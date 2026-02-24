"""Unit tests for knomi.cli."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from knomi.cli import app
from knomi.ingest.pipeline import PipelineResult

runner = CliRunner()


def test_ingest_help() -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "SOURCE_DIR" in result.output


def test_serve_help() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0


def test_status_help() -> None:
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_serve_starts_server() -> None:
    with patch("knomi.serve.server.start_server") as mock_start:
        result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    mock_start.assert_called_once()


def test_ingest_summarises_result(tmp_path: Path) -> None:
    mock_result = PipelineResult(total_files=3, total_chunks=12, total_vectors=12, skipped_files=1)
    with patch("knomi.ingest.pipeline.run_pipeline", return_value=mock_result):
        result = runner.invoke(app, ["ingest", str(tmp_path)])
    assert result.exit_code == 0
    assert "3 files" in result.output
    assert "12 chunks" in result.output
    assert "12 vectors" in result.output


def test_ingest_reports_skipped(tmp_path: Path) -> None:
    mock_result = PipelineResult(total_files=5, skipped_files=5)
    with patch("knomi.ingest.pipeline.run_pipeline", return_value=mock_result):
        result = runner.invoke(app, ["ingest", str(tmp_path)])
    assert result.exit_code == 0
    assert "skipped 5" in result.output


def test_status_prints_collection_info() -> None:
    mock_info: dict[str, object] = {
        "name": "knomi",
        "indexed_vectors_count": 10,
        "points_count": 10,
    }
    with patch("knomi.store.qdrant.QdrantStore") as mock_cls:
        mock_cls.return_value.describe.return_value = mock_info
        result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "knomi" in result.output
