"""Unit tests for knomi.serve.server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from knomi.config import Config
from knomi.ingest.chunker import Chunk
from knomi.serve.server import create_app
from knomi.store.base import SearchResult


def _search_results() -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(
                text="The capital of France is Paris.",
                metadata={"doc_id": "abc123", "chunk_index": 0, "source": "geo.txt"},
            ),
            score=0.92,
        ),
        SearchResult(
            chunk=Chunk(
                text="Paris is known for the Eiffel Tower.",
                metadata={"doc_id": "abc123", "chunk_index": 1, "source": "geo.txt"},
            ),
            score=0.87,
        ),
    ]


def _make_client(
    collection: str = "test-col",
) -> tuple[TestClient, MagicMock, MagicMock]:
    """Return a TestClient with mocked embedder and store."""
    config = Config(collection=collection)
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_store = MagicMock()
    mock_store.search.return_value = _search_results()
    with (
        patch("knomi.serve.server.build_embedder", return_value=mock_embedder),
        patch("knomi.serve.server.QdrantStore", return_value=mock_store),
    ):
        app = create_app(config)
    return TestClient(app), mock_embedder, mock_store


def test_health_returns_ok() -> None:
    client, _, _ = _make_client(collection="my-col")
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["collection"] == "my-col"


def test_query_returns_ranked_results() -> None:
    client, _, _ = _make_client()
    resp = client.post("/query", json={"query": "capital of France", "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["text"] == "The capital of France is Paris."
    assert data["results"][0]["score"] == 0.92
    assert data["collection"] == "test-col"


def test_query_calls_embedder_with_query_string() -> None:
    client, mock_embedder, _ = _make_client()
    client.post("/query", json={"query": "Who wrote Hamlet?", "top_k": 3})
    mock_embedder.embed_query.assert_called_once_with("Who wrote Hamlet?")


def test_query_passes_top_k_to_store() -> None:
    client, _, mock_store = _make_client()
    client.post("/query", json={"query": "test", "top_k": 7})
    call_args = mock_store.search.call_args
    assert call_args.kwargs["top_k"] == 7


def test_query_result_includes_metadata() -> None:
    client, _, _ = _make_client()
    resp = client.post("/query", json={"query": "Paris", "top_k": 1})
    result = resp.json()["results"][0]
    assert result["metadata"]["doc_id"] == "abc123"
    assert result["metadata"]["source"] == "geo.txt"


def test_query_missing_body_returns_422() -> None:
    client, _, _ = _make_client()
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_default_top_k_is_five() -> None:
    client, _, mock_store = _make_client()
    client.post("/query", json={"query": "test"})
    call_args = mock_store.search.call_args
    assert call_args.kwargs["top_k"] == 5


def test_openapi_schema_is_accessible() -> None:
    client, _, _ = _make_client()
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["info"]["title"] == "knomi RAG API"
    assert "/query" in schema["paths"]
    assert "/health" in schema["paths"]
