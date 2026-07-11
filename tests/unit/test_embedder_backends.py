"""Unit tests for build_embedder routing and the Cohere/Ollama backends.

The Cohere and Ollama SDK clients are monkeypatched with in-memory fakes so no
network calls or model downloads happen. OpenAI/local routing is verified by
patching their constructors.
"""

from __future__ import annotations

from unittest.mock import patch

import cohere
import ollama
import pytest

from knomi.config import Config
from knomi.ingest.embedder import (
    CohereEmbedder,
    LocalEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    build_embedder,
)


# --------------------------------------------------------------------------- #
# Fakes for the lazily-imported third-party clients.
# --------------------------------------------------------------------------- #
class _FakeCohereResponse:
    def __init__(self, texts: list[str]) -> None:
        self.embeddings = [[0.1, 0.2, 0.3] for _ in texts]


class _FakeCohereClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def embed(self, texts: list[str], model: str, input_type: str) -> _FakeCohereResponse:
        return _FakeCohereResponse(texts)


class _FakeOllamaClient:
    def __init__(self, host: str | None = None) -> None:
        self.host = host

    def embed(self, model: str, input: list[str]) -> dict[str, list[list[float]]]:
        return {"embeddings": [[0.4, 0.5] for _ in input]}


@pytest.fixture()
def fake_cohere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cohere, "Client", _FakeCohereClient, raising=True)


@pytest.fixture()
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "Client", _FakeOllamaClient, raising=True)


# --------------------------------------------------------------------------- #
# build_embedder routing
# --------------------------------------------------------------------------- #
def test_routes_to_openai() -> None:
    config = Config(embedding={"backend": "openai", "model": "text-embedding-3-small"})
    with patch("knomi.ingest.embedder.OpenAIEmbedder.__init__", return_value=None):
        embedder = build_embedder(config)
    assert isinstance(embedder, OpenAIEmbedder)


def test_routes_to_local() -> None:
    config = Config(embedding={"backend": "local", "model": "all-MiniLM-L6-v2"})
    with patch("knomi.ingest.embedder.LocalEmbedder.__init__", return_value=None):
        embedder = build_embedder(config)
    assert isinstance(embedder, LocalEmbedder)


def test_routes_to_cohere(fake_cohere: None) -> None:
    config = Config(embedding={"backend": "cohere", "model": "embed-english-v3.0"})
    embedder = build_embedder(config)
    assert isinstance(embedder, CohereEmbedder)


def test_routes_to_ollama(fake_ollama: None) -> None:
    config = Config(embedding={"backend": "ollama", "model": "nomic-embed-text"})
    embedder = build_embedder(config)
    assert isinstance(embedder, OllamaEmbedder)


# --------------------------------------------------------------------------- #
# Cohere / Ollama embed() behaviour with fakes
# --------------------------------------------------------------------------- #
def test_cohere_embed_returns_vectors(fake_cohere: None) -> None:
    embedder = CohereEmbedder("embed-english-v3.0", api_key="co-key")
    vectors = embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert all(isinstance(v, list) and len(v) == 3 for v in vectors)


def test_ollama_embed_returns_vectors(fake_ollama: None) -> None:
    embedder = OllamaEmbedder("nomic-embed-text", host="http://localhost:11434")
    vectors = embedder.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(isinstance(v, list) and len(v) == 2 for v in vectors)


def test_cohere_embed_query_returns_single_vector(fake_cohere: None) -> None:
    embedder = CohereEmbedder("embed-english-v3.0", api_key="co-key")
    vector = embedder.embed_query("just one")
    assert isinstance(vector, list)
    assert len(vector) == 3
