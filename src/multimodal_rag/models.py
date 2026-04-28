from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
import uuid


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source_path: str
    modality: Modality
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "modality": self.modality.value,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            source_path=str(payload["source_path"]),
            modality=Modality(str(payload["modality"])),
            content=str(payload["content"]),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    backend: str


@dataclass(slots=True)
class Citation:
    chunk_id: str
    source_path: str
    modality: Modality
    page_number: int | None = None
    excerpt: str | None = None


@dataclass(slots=True)
class QueryAnswer:
    answer: str
    hits: list[RetrievalHit]
    citations: list[Citation] = field(default_factory=list)
    retrieval_mode: str | None = None
    corrected: bool = False
    grounded: bool = True
    retrieval_diagnostics: dict[str, Any] = field(default_factory=dict)
    memory_context: str | None = None


MemoryEntityType = Literal["project", "preference", "fact", "task", "note"]


@dataclass(slots=True)
class MemoryNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    entity_type: MemoryEntityType = "fact"
    importance: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    access_count: int = 0
    relations: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    pinned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self, boost: float = 0.03) -> None:
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1
        self.importance = min(1.0, round(self.importance + boost, 4))

    def hours_since_access(self, now: datetime | None = None) -> float:
        reference = now or datetime.now(UTC)
        delta = reference - self.last_accessed
        return max(0.0, delta.total_seconds() / 3600)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "entity_type": self.entity_type,
            "importance": round(self.importance, 4),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "relations": list(self.relations),
            "embedding": list(self.embedding),
            "pinned": self.pinned,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryNode":
        data = dict(payload)
        for key in ("created_at", "last_accessed"):
            value = data.get(key)
            if isinstance(value, str):
                data[key] = datetime.fromisoformat(value)
        data.setdefault("metadata", {})
        data.setdefault("relations", [])
        data.setdefault("embedding", [])
        return cls(**data)
