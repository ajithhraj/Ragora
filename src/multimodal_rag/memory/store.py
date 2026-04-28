from __future__ import annotations

from pathlib import Path
import math
import re

import orjson

from multimodal_rag.config import Settings
from multimodal_rag.memory.decay import decay_node, should_prune
from multimodal_rag.memory.extractor import extract_memories
from multimodal_rag.models import MemoryNode

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class MemoryStore:
    def __init__(self, settings: Settings, embedder):
        self.settings = settings
        self.embedder = embedder
        self.base_dir = settings.memory_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(raw: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw.strip().lower())
        safe = safe.strip("-_")
        return safe or "default"

    def _scope_path(self, tenant_id: str, session_id: str) -> Path:
        tenant = self._safe_name(tenant_id)
        session = self._safe_name(session_id)
        return self.base_dir / tenant / f"{session}.json"

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _load(self, tenant_id: str, session_id: str) -> list[MemoryNode]:
        path = self._scope_path(tenant_id, session_id)
        if not path.exists():
            return []
        payload = orjson.loads(path.read_bytes())
        return [MemoryNode.from_payload(item) for item in payload]

    def _save(self, tenant_id: str, session_id: str, nodes: list[MemoryNode]) -> None:
        path = self._scope_path(tenant_id, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [node.to_payload() for node in nodes]
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    def _apply_decay(self, tenant_id: str, session_id: str, nodes: list[MemoryNode]) -> list[MemoryNode]:
        kept: list[MemoryNode] = []
        changed = False
        for node in nodes:
            before = node.importance
            decay_node(
                node,
                rate=self.settings.memory_decay_rate,
                pinned_floor=self.settings.memory_pinned_floor,
            )
            if should_prune(node, threshold=self.settings.memory_prune_threshold):
                changed = True
                continue
            if node.importance != before:
                changed = True
            kept.append(node)
        if changed:
            self._save(tenant_id, session_id, kept)
        return kept

    def _refresh_relations(self, nodes: list[MemoryNode]) -> None:
        token_sets = {node.id: self._token_set(node.content) for node in nodes}
        for node in nodes:
            node_tokens = token_sets[node.id]
            related: list[str] = []
            for other in nodes:
                if other.id == node.id:
                    continue
                overlap = len(node_tokens & token_sets[other.id])
                if overlap >= 2:
                    related.append(other.id)
            node.relations = related[:8]

    def remember(
        self,
        message: str,
        *,
        tenant_id: str,
        session_id: str,
        pinned: bool = False,
        metadata: dict | None = None,
    ) -> list[MemoryNode]:
        nodes = self._load(tenant_id, session_id)
        extracted = extract_memories(message, force_pinned=pinned)
        if not extracted:
            return []

        existing_by_content = {node.content.lower(): node for node in nodes}
        remembered: list[MemoryNode] = []
        new_nodes: list[MemoryNode] = []
        for node in extracted:
            if metadata:
                node.metadata.update(metadata)
            key = node.content.lower()
            existing = existing_by_content.get(key)
            if existing:
                existing.importance = max(existing.importance, node.importance)
                existing.pinned = existing.pinned or node.pinned
                existing.touch(boost=0.06 if existing.pinned else 0.03)
                existing.metadata.update(node.metadata)
                remembered.append(existing)
            else:
                new_nodes.append(node)
                remembered.append(node)

        if new_nodes:
            vectors = self.embedder.embed_documents([node.content for node in new_nodes])
            for node, vector in zip(new_nodes, vectors, strict=False):
                node.embedding = list(vector)
                nodes.append(node)

        self._refresh_relations(nodes)
        self._save(tenant_id, session_id, nodes)
        return remembered

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        session_id: str,
        top_k: int | None = None,
    ) -> list[MemoryNode]:
        nodes = self._apply_decay(tenant_id, session_id, self._load(tenant_id, session_id))
        if not nodes:
            return []

        query_vector = self.embedder.embed_query(query)
        query_tokens = self._token_set(query)
        ranked: list[tuple[float, MemoryNode]] = []
        for node in nodes:
            node_tokens = self._token_set(node.content)
            overlap = len(query_tokens & node_tokens)
            graph_relevance = overlap / max(1, len(query_tokens)) if query_tokens else 0.0
            vector_similarity = self._cosine_similarity(query_vector, node.embedding)
            recency_bonus = 1.0 / (1.0 + node.hours_since_access())
            score = (0.45 * node.importance) + (0.25 * graph_relevance) + (0.2 * vector_similarity) + (0.1 * recency_bonus)
            ranked.append((score, node))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [node for _, node in ranked[: top_k or self.settings.memory_top_k] if _ > 0.0]
        for node in selected:
            node.touch()
        if selected:
            self._save(tenant_id, session_id, nodes)
        return selected

    def build_context(
        self,
        query: str,
        *,
        tenant_id: str,
        session_id: str,
        top_k: int | None = None,
    ) -> str:
        nodes = self.retrieve(query, tenant_id=tenant_id, session_id=session_id, top_k=top_k)
        if not nodes:
            return ""
        lines = ["[MEMORY CONTEXT]", "Use these session memories only when relevant and consistent with the retrieved sources:"]
        for index, node in enumerate(nodes, start=1):
            lines.append(
                f"{index}. [{node.entity_type.upper()}] {node.content} "
                f"(importance={node.importance:.2f}, pinned={'yes' if node.pinned else 'no'})"
            )
        return "\n".join(lines)

    def export(self, *, tenant_id: str, session_id: str) -> list[MemoryNode]:
        return self._apply_decay(tenant_id, session_id, self._load(tenant_id, session_id))

    def stats(self, *, tenant_id: str, session_id: str) -> dict[str, float | int]:
        nodes = self.export(tenant_id=tenant_id, session_id=session_id)
        if not nodes:
            return {"count": 0, "pinned_count": 0, "avg_importance": 0.0}
        avg = sum(node.importance for node in nodes) / len(nodes)
        pinned = sum(1 for node in nodes if node.pinned)
        return {"count": len(nodes), "pinned_count": pinned, "avg_importance": round(avg, 4)}

    def forget(self, memory_id: str, *, tenant_id: str, session_id: str) -> bool:
        nodes = self._load(tenant_id, session_id)
        kept = [node for node in nodes if node.id != memory_id]
        if len(kept) == len(nodes):
            return False
        self._refresh_relations(kept)
        self._save(tenant_id, session_id, kept)
        return True

    def reinforce(self, memory_id: str, *, tenant_id: str, session_id: str) -> MemoryNode | None:
        nodes = self._load(tenant_id, session_id)
        for node in nodes:
            if node.id == memory_id:
                node.pinned = True
                node.touch(boost=0.2)
                node.importance = min(1.0, round(node.importance + 0.2, 4))
                self._save(tenant_id, session_id, nodes)
                return node
        return None
