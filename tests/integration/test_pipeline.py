"""Integration tests for the ingest pipeline.

Requires a running Qdrant instance (set KNOMI_DB_URL or use the default
http://localhost:6333). Run with: pytest tests/integration -v
"""

from __future__ import annotations

from pathlib import Path

from knomi.config import Config
from knomi.ingest.pipeline import run_pipeline


def _write_minimal_pdf(path: Path, text: str = "PDF document content for testing.") -> None:
    """Write a minimal valid PDF to *path* using PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_ingest_pdf_creates_vectors(tmp_path: Path, qdrant_url: str) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_minimal_pdf(pdf_path, "Integration test PDF content for embedding.")
    config = Config(
        source_dir=tmp_path,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
        db_url=qdrant_url,
        collection="test-ingest-pipeline",
        chunk_size=256,
        chunk_overlap=32,
    )
    result = run_pipeline(config)
    assert result.total_files >= 1
    assert result.total_vectors > 0
    assert len(result.failed_files) == 0


def test_ingest_dedup_skips_on_second_run(tmp_path: Path, qdrant_url: str) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_minimal_pdf(pdf_path, "Dedup test PDF content.")
    config = Config(
        source_dir=tmp_path,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
        db_url=qdrant_url,
        collection="test-ingest-dedup",
        chunk_size=256,
        chunk_overlap=32,
    )
    first = run_pipeline(config)
    assert first.total_files >= 1

    second = run_pipeline(config)
    assert second.skipped_files == first.total_files
    assert second.total_vectors == 0


def test_serve_query_returns_results(tmp_path: Path, qdrant_url: str) -> None:
    from fastapi.testclient import TestClient

    from knomi.serve.server import create_app

    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("The quick brown fox jumps over the lazy dog. " * 10)
    ingest_config = Config(
        source_dir=tmp_path,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
        db_url=qdrant_url,
        collection="test-serve-query",
        chunk_size=64,
        chunk_overlap=8,
    )
    run_pipeline(ingest_config)

    serve_config = Config(
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
        db_url=qdrant_url,
        collection="test-serve-query",
    )
    client = TestClient(create_app(serve_config))
    response = client.post("/query", json={"query": "quick brown fox", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) >= 1


def test_delete_removes_vectors(tmp_path: Path, qdrant_url: str) -> None:
    from knomi.ingest.scanner import scan
    from knomi.store.qdrant import QdrantStore

    txt_path = tmp_path / "delete_me.txt"
    txt_path.write_text("Document that will be deleted after indexing. " * 5)
    config = Config(
        source_dir=tmp_path,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
        db_url=qdrant_url,
        collection="test-delete-vectors",
        chunk_size=64,
        chunk_overlap=8,
    )
    result = run_pipeline(config)
    assert result.total_files == 1
    assert result.total_vectors > 0

    scanned_files = list(scan(tmp_path))
    assert len(scanned_files) == 1
    doc_id = scanned_files[0].sha256

    store = QdrantStore(config)
    assert store.has_document(doc_id)

    store.delete(doc_id)
    assert not store.has_document(doc_id)
