from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

SECTION_NUMBER_RE = re.compile(r"^(\d+(\.\d+){0,3}|[IVXLCM]+)[\)\.\-:\s]+[A-Za-z]")


def split_text(content: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = content.strip()
    if not clean:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [piece.strip() for piece in splitter.split_text(clean) if piece.strip()]


def looks_like_heading(line: str) -> bool:
    text = line.strip()
    if len(text) < 3 or len(text) > 120:
        return False
    if text.endswith((".", "?", "!")):
        return False

    words = [w for w in text.split() if w]
    if not words or len(words) > 14:
        return False

    if SECTION_NUMBER_RE.match(text):
        return True

    alpha_chars = [ch for ch in text if ch.isalpha()]
    if alpha_chars:
        uppercase_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
        if uppercase_ratio >= 0.75:
            return True

    titled_words = 0
    for word in words:
        stripped = word.strip("()[]{}:;,.-_")
        if stripped and stripped[0].isupper():
            titled_words += 1
    if len(words) <= 8 and titled_words / len(words) >= 0.8:
        return True

    return False


def split_structured_segments(
    segments: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Split already segmented sections while preserving section metadata."""
    output: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        base_metadata = dict(segment.get("metadata", {}) or {})
        parts = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for part_index, part in enumerate(parts, start=1):
            output.append(
                {
                    "text": part,
                    "metadata": {
                        **base_metadata,
                        "segment_part": part_index,
                    },
                }
            )
    return output
