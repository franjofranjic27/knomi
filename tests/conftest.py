"""Shared pytest fixtures.

Fixtures
--------
tmp_docs_dir    tmp_path pre-populated with sample .txt and placeholder .pdf files.
mock_embedder   A BaseEmbedder that returns zero vectors without loading a model.
qdrant_url      Reads KNOMI_DB_URL env var or falls back to http://localhost:6333.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from knomi.ingest.embedder import BaseEmbedder


@pytest.fixture()
def tmp_docs_dir(tmp_path: Path) -> Path:
    """Return a temp directory with a minimal set of sample documents.

    TODO:
        - Write a small .txt file with known content.
        - Write a minimal valid PDF (or copy a fixture PDF from tests/fixtures/).
        - Write a .md file.
    """
    (tmp_path / "sample.txt").write_text("Hello world. This is a test document.")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("Nested document content.")
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
