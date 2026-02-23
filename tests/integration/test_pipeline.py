"""Integration tests for the ingest pipeline.

Requires a running Qdrant instance (set KNOMI_DB_URL or use the default
http://localhost:6333). Run with: pytest tests/integration -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knomi.config import Config
from knomi.ingest.pipeline import run_pipeline

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PDF_PATH = DATA_DIR / "Bandi2025_AgenticAI_Rise-of-Agentic-AI.pdf"


@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF fixture not present")
def test_ingest_pdf_creates_vectors(qdrant_url: str) -> None:
    config = Config(
        source_dir=DATA_DIR,
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


@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF fixture not present")
def test_ingest_dedup_skips_on_second_run(qdrant_url: str) -> None:
    config = Config(
        source_dir=DATA_DIR,
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
