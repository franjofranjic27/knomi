"""Shared pytest fixtures.

Fixtures
--------
tmp_docs_dir    tmp_path pre-populated with sample .txt and .pdf files.
pdf_file        A minimal valid PDF with known text content.
mock_embedder   A BaseEmbedder that returns zero vectors without loading a model.
qdrant_url      Reads KNOMI_DB_URL env var or falls back to http://localhost:6333.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from knomi.ingest.embedder import BaseEmbedder


def _write_minimal_pdf(path: Path, text: str = "PDF document content for testing.") -> None:
    """Write a minimal valid PDF to *path* using PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    """Return a path to a minimal valid PDF with known text content."""
    path = tmp_path / "sample.pdf"
    _write_minimal_pdf(path, "Hello PDF world. This is a test document.")
    return path


@pytest.fixture()
def tmp_docs_dir(tmp_path: Path) -> Path:
    """Return a temp directory with a minimal set of sample documents."""
    (tmp_path / "sample.txt").write_text("Hello world. This is a test document.")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("Nested document content.")
    _write_minimal_pdf(tmp_path / "sample.pdf")
    return tmp_path


class _MockEmbedder(BaseEmbedder):
    """Returns zero vectors of dimension 4 without any model loading."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture()
def mock_embedder() -> BaseEmbedder:
    """Embedder stub that returns fixed-dimension zero vectors."""
    return _MockEmbedder()


@pytest.fixture()
def qdrant_url() -> str:
    """Qdrant server URL for integration tests."""
    return os.environ.get("KNOMI_DB_URL", "http://localhost:6333")
