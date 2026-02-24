"""Unit tests for knomi.ingest.embedder."""

from unittest.mock import patch

from knomi.config import Config
from knomi.ingest.chunker import Chunk
from knomi.ingest.embedder import (
    BaseEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    build_embedder,
)


def _make_chunks(n: int) -> list[Chunk]:
    return [Chunk(text=f"chunk {i}", metadata={"doc_id": "x", "chunk_index": i}) for i in range(n)]


def test_build_embedder_openai_routing() -> None:
    config = Config(embedding_model="text-embedding-3-small")
    with patch("knomi.ingest.embedder.OpenAIEmbedder.__init__", return_value=None):
        embedder = build_embedder(config)
    assert isinstance(embedder, OpenAIEmbedder)


def test_build_embedder_local_routing() -> None:
    config = Config(embedding_model="all-MiniLM-L6-v2")
    with patch("knomi.ingest.embedder.LocalEmbedder.__init__", return_value=None):
        embedder = build_embedder(config)
    assert isinstance(embedder, LocalEmbedder)


def test_embed_chunks_batching(mock_embedder: BaseEmbedder) -> None:
    chunks = _make_chunks(10)
    vectors = mock_embedder.embed_chunks(chunks, batch_size=3)
    assert len(vectors) == 10
    assert all(len(v) == 4 for v in vectors)


def test_embed_chunks_single_batch(mock_embedder: BaseEmbedder) -> None:
    chunks = _make_chunks(5)
    vectors = mock_embedder.embed_chunks(chunks, batch_size=100)
    assert len(vectors) == 5


def test_embed_chunks_empty(mock_embedder: BaseEmbedder) -> None:
    assert mock_embedder.embed_chunks([], batch_size=10) == []


def test_embed_query_returns_single_vector(mock_embedder: BaseEmbedder) -> None:
    vector = mock_embedder.embed_query("what is RAG?")
    assert isinstance(vector, list)
    assert len(vector) == 4


def test_embed_chunks_concurrent(mock_embedder: BaseEmbedder) -> None:
    chunks = _make_chunks(9)
    sequential = mock_embedder.embed_chunks(chunks, batch_size=3, workers=1)
    concurrent = mock_embedder.embed_chunks(chunks, batch_size=3, workers=3)
    assert sequential == concurrent
    assert len(concurrent) == 9
