from pathlib import Path

from multimodal_rag.config import Settings
from multimodal_rag.memory.store import MemoryStore


class DummyEmbedder:
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        lowered = text.lower()
        return [
            1.0 if "concise" in lowered else 0.0,
            1.0 if "rag" in lowered else 0.0,
            float(len(lowered.split())) / 10.0,
        ]


def test_memory_store_remembers_and_retrieves(tmp_path):
    settings = Settings(storage_dir=Path(tmp_path), memory_dir=Path(tmp_path) / "memory")
    store = MemoryStore(settings, DummyEmbedder())

    remembered = store.remember(
        "remember I prefer concise answers when we discuss the RAG app.",
        tenant_id="public",
        session_id="demo",
        pinned=True,
    )
    assert len(remembered) == 1
    assert remembered[0].pinned is True

    nodes = store.retrieve("be concise about rag", tenant_id="public", session_id="demo")
    assert len(nodes) == 1
    assert "concise answers" in nodes[0].content.lower()


def test_memory_store_skips_generic_questions(tmp_path):
    settings = Settings(storage_dir=Path(tmp_path), memory_dir=Path(tmp_path) / "memory")
    store = MemoryStore(settings, DummyEmbedder())

    remembered = store.remember(
        "What are the key risks in this document?",
        tenant_id="public",
        session_id="demo",
    )
    assert remembered == []
