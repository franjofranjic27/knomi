"""Unit tests for knomi.config (nested settings groups)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from knomi.config import Config, resolve_config

# Environment variables that can leak into Config from the shell or a repo .env
# file and must be cleared for deterministic assertions.
_LEAKY_ENV = [
    "KNOMI_PROFILE",
    "KNOMI_CHUNKING__CHUNK_SIZE",
    "KNOMI_CHUNKING__CHUNK_OVERLAP",
    "KNOMI_STORE__COLLECTION",
    "KNOMI_STORE__URL",
    "KNOMI_EMBEDDING__MODEL",
    "KNOMI_EMBEDDING__DIM",
    "KNOMI_EMBEDDING__BATCH_SIZE",
    "KNOMI_EMBEDDING__API_KEY",
    "KNOMI_SERVE_HOST",
    "KNOMI_SERVE_PORT",
    "KNOMI_TOP_K",
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "QDRANT_API_KEY",
    "KNOMI_PG_DSN",
]


@pytest.fixture()
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Chdir to a clean dir (no .env / knomi.json) and drop leaky env vars."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for key in _LEAKY_ENV:
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def test_default_values(isolated_env: Path) -> None:
    config = Config()
    assert config.chunking.chunk_size == 512
    assert config.chunking.chunk_overlap == 64
    assert config.chunking.strategy == "token"
    assert config.embedding.backend == "openai"
    assert config.embedding.model == "text-embedding-3-small"
    assert config.embedding.dim == 1536
    assert config.embedding.batch_size == 64
    assert config.store.backend == "qdrant"
    assert config.store.url == "http://localhost:6333"
    assert config.store.collection == "knomi"
    assert config.serve_host == "0.0.0.0"
    assert config.serve_port == 8080
    assert config.top_k == 5
    assert config.embedding.api_key is None


def test_explicit_constructor_overrides_defaults() -> None:
    config = Config(
        chunking={"chunk_size": 256, "chunk_overlap": 32},
        store={"collection": "my-kb"},
    )
    assert config.chunking.chunk_size == 256
    assert config.chunking.chunk_overlap == 32
    assert config.store.collection == "my-kb"


def test_env_var_overrides_default(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOMI_CHUNKING__CHUNK_SIZE", "128")
    config = Config()
    assert config.chunking.chunk_size == 128


def test_env_var_collection(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOMI_STORE__COLLECTION", "custom-col")
    config = Config()
    assert config.store.collection == "custom-col"


def test_env_var_embedding_dim(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOMI_EMBEDDING__DIM", "384")
    config = Config()
    assert config.embedding.dim == 384


def test_openai_api_key_from_standard_env_var(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-standard")
    config = resolve_config()
    assert config.embedding.api_key == "sk-test-standard"


def test_embedding_api_key_from_nested_env_var(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KNOMI_EMBEDDING__API_KEY", "sk-test-knomi")
    config = Config()
    assert config.embedding.api_key == "sk-test-knomi"


def test_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Config(chunking={"chunk_size": 0})


def test_chunk_overlap_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        Config(chunking={"chunk_overlap": -1})


def test_embedding_dim_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Config(embedding={"dim": 0})


def test_serve_port_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Config(serve_port=70000)


def test_source_dir_accepts_path_object(tmp_path: Path) -> None:
    config = Config(source_dir=tmp_path)
    assert config.source_dir == tmp_path


def test_constructor_takes_priority_over_env_var(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KNOMI_CHUNKING__CHUNK_SIZE", "999")
    config = Config(chunking={"chunk_size": 42})
    assert config.chunking.chunk_size == 42
