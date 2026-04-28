import pytest


@pytest.fixture(autouse=True)
def _reset_runtime_env(monkeypatch):
    monkeypatch.setenv("MMRAG_OPENAI_API_KEY", "")
    monkeypatch.setenv("MMRAG_API_ONLY_MODE", "false")
