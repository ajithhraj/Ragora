from __future__ import annotations

import math

from multimodal_rag.models import MemoryNode


def decay_node(
    node: MemoryNode,
    *,
    rate: float,
    pinned_floor: float,
) -> MemoryNode:
    hours = node.hours_since_access()
    new_importance = node.importance * math.exp(-rate * hours)
    if node.pinned:
        new_importance = max(new_importance, pinned_floor)
    node.importance = round(max(0.0, new_importance), 4)
    return node


def should_prune(node: MemoryNode, *, threshold: float) -> bool:
    return node.importance < threshold and not node.pinned
