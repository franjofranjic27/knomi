"""End-to-end unit tests for the ChromaStore backend.

Chroma runs fully in-process with on-disk persistence in ``tmp_path`` (no Docker,
no network), so this exercises the real store implementation quickly and
deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knomi.config import Config
from knomi.ingest.chunker import Chunk
from knomi.store.chroma import ChromaStore

_DIM = 4


def _config(tmp_path: Path, collection: str = "unit-kb") -> Config:
    return Config(
        store={"backend": "chroma", "path": str(tmp_path / "chroma"), "collection": collection},
        embedding={"backend": "local", "dim": _DIM},
    )


def _chunk(doc_id: str, index: int, total: int, source: str, vec: list[float]) -> Chunk:
    return Chunk(
        text=f"chunk {index} of {doc_id}",
        metadata={
            "doc_id": doc_id,
            "chunk_index": index,
            "total_chunks": total,
            "source": source,
        },
    )


@pytest.fixture()
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(_config(tmp_path))


def _seed(store: ChromaStore) -> str:
    doc_id = "doc-aaa"
    chunks = [
        _chunk(doc_id, 0, 2, "/docs/a.txt", [1.0, 0.0, 0.0, 0.0]),
        _chunk(doc_id, 1, 2, "/docs/a.txt", [0.0, 1.0, 0.0, 0.0]),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    store.upsert(chunks, vectors)
    return doc_id


def test_upsert_then_has_document(store: ChromaStore) -> None:
    doc_id = _seed(store)
    assert store.has_document(doc_id)
    assert not store.has_document("missing-doc")


def test_search_returns_nearest_chunk(store: ChromaStore) -> None:
    _seed(store)
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    # The vector identical to the first chunk should rank first.
    assert results[0].chunk.text == "chunk 0 of doc-aaa"
    assert results[0].score >= results[1].score


def test_get_source_returns_stored_path(store: ChromaStore) -> None:
    doc_id = _seed(store)
    assert store.get_source(doc_id) == "/docs/a.txt"
    assert store.get_source("missing-doc") is None


def test_update_source_rewrites_all_chunks(store: ChromaStore) -> None:
    doc_id = _seed(store)
    store.update_source(doc_id, "/moved/a.txt")
    assert store.get_source(doc_id) == "/moved/a.txt"


def test_describe_reports_point_count(store: ChromaStore) -> None:
    _seed(store)
    info = store.describe()
    assert info["name"] == "unit-kb"
    assert info["points_count"] == 2


def test_delete_removes_document(store: ChromaStore) -> None:
    doc_id = _seed(store)
    store.delete(doc_id)
    assert not store.has_document(doc_id)
    assert store.describe()["points_count"] == 0


def test_upsert_is_idempotent(store: ChromaStore) -> None:
    _seed(store)
    _seed(store)  # same deterministic ids → no duplicates
    assert store.describe()["points_count"] == 2


def test_upsert_empty_is_noop(store: ChromaStore) -> None:
    store.upsert([], [])
    assert store.describe()["points_count"] == 0
