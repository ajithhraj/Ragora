import pytest

from multimodal_rag.config import Settings
from multimodal_rag.embedding.providers import TextEmbedder


def test_settings_strict_api_only_mode_for_openai_provider(tmp_path):
    settings = Settings(storage_dir=tmp_path, llm_provider="openai", openai_api_key="test-key")
    assert settings.strict_api_only_mode() is True


def test_settings_strict_api_only_mode_for_explicit_flag(tmp_path):
    settings = Settings(storage_dir=tmp_path, llm_provider="ollama", api_only_mode=True)
    assert settings.strict_api_only_mode() is True


def test_text_embedder_raises_without_openai_key_in_strict_mode(tmp_path):
    settings = Settings(storage_dir=tmp_path, llm_provider="openai", openai_api_key=None, api_only_mode=True)
    with pytest.raises(RuntimeError, match="OpenAI API key is required"):
        TextEmbedder(settings)
