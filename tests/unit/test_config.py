"""Unit tests for knomi.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from knomi.config import Config


def test_default_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # chdir to a directory with no .env file to isolate from repo-level .env overrides
    monkeypatch.chdir(tmp_path)
    for key in [
        "KNOMI_CHUNK_SIZE",
        "KNOMI_CHUNK_OVERLAP",
        "KNOMI_COLLECTION",
        "KNOMI_DB_URL",
        "KNOMI_EMBEDDING_MODEL",
        "KNOMI_EMBEDDING_DIM",
        "KNOMI_EMBEDDING_BATCH_SIZE",
        "KNOMI_SERVE_HOST",
        "KNOMI_SERVE_PORT",
        "KNOMI_TOP_K",
        "OPENAI_API_KEY",
        "KNOMI_OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    config = Config()
    assert config.chunk_size == 512
    assert config.chunk_overlap == 64
    assert config.embedding_model == "text-embedding-3-small"
    assert config.embedding_dim == 1536
    assert config.embedding_batch_size == 64
    assert config.db_url == "http://localhost:6333"
    assert config.collection == "knomi"
    assert config.serve_host == "0.0.0.0"
    assert config.serve_port == 8080
    assert config.top_k == 5
    assert config.openai_api_key is None


def test_explicit_constructor_overrides_defaults() -> None:
    config = Config(chunk_size=256, chunk_overlap=32, collection="my-kb")
    assert config.chunk_size == 256
    assert config.chunk_overlap == 32
    assert config.collection == "my-kb"


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOMI_CHUNK_SIZE", "128")
    config = Config()
    assert config.chunk_size == 128


def test_env_var_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOMI_COLLECTION", "custom-col")
    config = Config()
    assert config.collection == "custom-col"


def test_openai_api_key_from_standard_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-standard")
    config = Config()
    assert config.openai_api_key == "sk-test-standard"


def test_openai_api_key_from_knomi_prefixed_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Use a clean directory (no .env) so OPENAI_API_KEY from .env doesn't conflict
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("KNOMI_OPENAI_API_KEY", "sk-test-knomi")
    config = Config()
    assert config.openai_api_key == "sk-test-knomi"


def test_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Config(chunk_size=0)


def test_chunk_overlap_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        Config(chunk_overlap=-1)


def test_serve_port_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Config(serve_port=70000)


def test_source_dir_accepts_path_object(tmp_path: Path) -> None:
    config = Config(source_dir=tmp_path)
    assert config.source_dir == tmp_path


def test_constructor_takes_priority_over_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOMI_CHUNK_SIZE", "999")
    config = Config(chunk_size=42)
    assert config.chunk_size == 42
