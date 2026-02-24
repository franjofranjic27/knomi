"""Unit tests for knomi.ingest.pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from knomi.config import Config
from knomi.ingest.chunker import Chunk
from knomi.ingest.embedder import BaseEmbedder
from knomi.ingest.pipeline import PipelineResult, run_pipeline
from knomi.ingest.scanner import _sha256
from knomi.store.base import SearchResult, VectorStore


class _MockStore(VectorStore):
    """In-memory VectorStore stub for pipeline unit tests."""

    def __init__(self, indexed: dict[str, str] | None = None) -> None:
        # indexed maps sha256 → stored source path
        self._indexed: dict[str, str] = indexed or {}
        self.upserted: list[tuple[list[Chunk], list[list[float]]]] = []
        self.updated: dict[str, str] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.upserted.append((chunks, vectors))

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        return []

    def delete(self, doc_id: str) -> None:
        self._indexed.pop(doc_id, None)

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self._indexed

    def describe(self) -> dict[str, object]:
        return {"name": "mock", "points_count": len(self._indexed)}

    def get_source(self, doc_id: str) -> str | None:
        return self._indexed.get(doc_id)

    def update_source(self, doc_id: str, new_source: str) -> None:
        self.updated[doc_id] = new_source


class _MockEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _run(source_dir: Path, store: _MockStore) -> PipelineResult:
    config = Config(source_dir=source_dir, embedding_dim=4)
    with (
        patch("knomi.store.qdrant.QdrantStore", return_value=store),
        patch("knomi.ingest.embedder.build_embedder", return_value=_MockEmbedder()),
    ):
        return run_pipeline(config)


def test_pipeline_processes_text_files(tmp_docs_dir: Path) -> None:
    store = _MockStore()
    result = _run(tmp_docs_dir, store)
    assert result.total_files >= 1
    assert len(result.failed_files) == 0
    assert len(store.upserted) >= 1


def test_pipeline_result_tracks_vectors(tmp_docs_dir: Path) -> None:
    store = _MockStore()
    result = _run(tmp_docs_dir, store)
    assert result.total_vectors > 0
    assert result.total_chunks == result.total_vectors


def test_pipeline_skips_already_indexed(tmp_path: Path) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("some content that will be indexed")
    sha = _sha256(doc)
    store = _MockStore(indexed={sha: str(doc)})
    result = _run(tmp_path, store)
    assert result.skipped_files == 1
    assert store.upserted == []  # nothing re-embedded


def test_pipeline_detects_moved_file(tmp_docs_dir: Path) -> None:
    sample = tmp_docs_dir / "sample.txt"
    sha = _sha256(sample)
    old_path = "/old/location/sample.txt"
    store = _MockStore(indexed={sha: old_path})
    result = _run(tmp_docs_dir, store)
    assert result.moved_files >= 1
    assert store.updated[sha] == str(sample)


def test_pipeline_move_still_counts_as_skipped(tmp_docs_dir: Path) -> None:
    sample = tmp_docs_dir / "sample.txt"
    sha = _sha256(sample)
    store = _MockStore(indexed={sha: "/old/path.txt"})
    result = _run(tmp_docs_dir, store)
    # moved file is also counted as skipped (not re-embedded)
    assert result.skipped_files >= 1
    assert result.total_vectors == 0 or result.total_vectors == result.total_chunks


def test_pipeline_empty_text_counts_as_failed(tmp_path: Path) -> None:
    # An unsupported extension produces empty text → pipeline marks it as failed
    (tmp_path / "empty.txt").write_text("")
    store = _MockStore()
    result = _run(tmp_path, store)
    assert len(result.failed_files) >= 1
