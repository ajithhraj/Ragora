from __future__ import annotations

import re

from multimodal_rag.models import MemoryEntityType, MemoryNode

EXPLICIT_MEMORY_RE = re.compile(r"^\s*!remember\b|^\s*remember\s*[:\-]?\s*", flags=re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
FIRST_PERSON_RE = re.compile(r"\b(i am|i'm|i work|i build|i prefer|i like|i need|my |we are|we're|our )", re.I)
PREFERENCE_RE = re.compile(r"\b(prefer|like|want|need|should|must|always|never)\b", re.I)
TASK_RE = re.compile(r"\b(build|implement|ship|deploy|finish|working on|need to|todo|task)\b", re.I)
QUESTION_RE = re.compile(r"\?\s*$")


def _normalize_sentence(text: str) -> str:
    compact = " ".join(text.strip().split())
    return compact.strip(" -")


def _infer_entity_type(text: str) -> MemoryEntityType:
    lowered = text.lower()
    if PREFERENCE_RE.search(lowered):
        return "preference"
    if TASK_RE.search(lowered):
        return "task"
    if FIRST_PERSON_RE.search(lowered):
        return "project"
    if len(lowered.split()) <= 4:
        return "note"
    return "fact"


def _base_importance(entity_type: MemoryEntityType, pinned: bool) -> float:
    if pinned:
        return 0.95
    return {
        "preference": 0.84,
        "task": 0.78,
        "project": 0.74,
        "fact": 0.65,
        "note": 0.58,
    }[entity_type]


def extract_memories(message: str, *, force_pinned: bool = False) -> list[MemoryNode]:
    content = message.strip()
    if not content:
        return []

    explicit_memory = bool(EXPLICIT_MEMORY_RE.search(content))
    content = EXPLICIT_MEMORY_RE.sub("", content).strip()
    parts = [_normalize_sentence(part) for part in SENTENCE_SPLIT_RE.split(content)]

    nodes: list[MemoryNode] = []
    for part in parts:
        if not part:
            continue
        pinned = force_pinned or explicit_memory
        if QUESTION_RE.search(part) and not pinned:
            continue
        if len(part) < 8:
            continue
        if not pinned and not (FIRST_PERSON_RE.search(part) or PREFERENCE_RE.search(part) or TASK_RE.search(part)):
            continue
        entity_type = _infer_entity_type(part)
        nodes.append(
            MemoryNode(
                content=part,
                entity_type=entity_type,
                importance=_base_importance(entity_type, pinned),
                pinned=pinned,
            )
        )
    return nodes
