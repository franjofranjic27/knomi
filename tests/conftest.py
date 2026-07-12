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

# knomi.json / secret env vars that must not leak into tests from the developer's
# shell or a knomi.json sitting in the repo root.
_PROFILE_ENV = (
    "KNOMI_PROFILE",
    "KNOMI_STORE__COLLECTION",
    "KNOMI_STORE__BACKEND",
    "KNOMI_EMBEDDING__DIM",
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "QDRANT_API_KEY",
    "KNOMI_PG_DSN",
)


@pytest.fixture(autouse=True)
def _isolate_ambient_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run every test in a clean CWD with profile-related env vars cleared.

    Without this, a ``knomi.json`` in the working directory (e.g. one a developer
    created for a real run) or an exported ``KNOMI_*`` / API-key variable would
    silently seep into ``Config`` and change what backends/strategies tests see.
    Tests that exercise profiles write their own ``knomi.json`` and chdir
    explicitly, which overrides this isolation.
    """
    clean_cwd = tmp_path / "_clean_cwd"
    clean_cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(clean_cwd)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "_xdg"))
    for key in _PROFILE_ENV:
        monkeypatch.delenv(key, raising=False)


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
