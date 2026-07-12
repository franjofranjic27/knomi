"""Unit tests for knomi.json profile loading and resolution precedence.

These tests isolate the profile file lookup by chdir-ing into ``tmp_path`` and
pointing ``XDG_CONFIG_HOME`` at an empty directory, so neither the repository's
own config nor the developer's ``~/.config`` can leak in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knomi.config import load_profiles, resolve_config, resolve_profile_name

# knomi.json / secret env vars that must not leak into these assertions.
_LEAKY_ENV = [
    "KNOMI_PROFILE",
    "KNOMI_STORE__COLLECTION",
    "KNOMI_STORE__BACKEND",
    "KNOMI_EMBEDDING__DIM",
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "QDRANT_API_KEY",
    "KNOMI_PG_DSN",
]

_PROFILE_DOC = {
    "default_profile": "local",
    "profiles": {
        "local": {
            "store": {"backend": "chroma", "path": "/data/chroma", "collection": "kb-local"},
            "embedding": {"backend": "local", "model": "all-MiniLM-L6-v2", "dim": 384},
            "chunking": {"strategy": "structure", "chunk_size": 256, "chunk_overlap": 32},
        },
        "cloud": {
            "store": {"backend": "qdrant", "url": "https://c.qdrant.io", "collection": "kb-cloud"},
            "embedding": {"backend": "openai", "model": "text-embedding-3-small", "dim": 1536},
            "chunking": {"strategy": "token", "chunk_size": 512, "chunk_overlap": 64},
        },
    },
}


@pytest.fixture()
def profiles_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Chdir into a temp dir containing a knomi.json and a clean environment."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for key in _LEAKY_ENV:
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "knomi.json").write_text(json.dumps(_PROFILE_DOC), encoding="utf-8")
    return tmp_path


def test_load_profiles_reads_document(profiles_dir: Path) -> None:
    data = load_profiles()
    assert data["default_profile"] == "local"
    assert set(data["profiles"]) == {"local", "cloud"}


def test_load_profiles_absent_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert load_profiles() == {}


def test_resolve_profile_name_prefers_explicit(profiles_dir: Path) -> None:
    assert resolve_profile_name("cloud") == "cloud"


def test_resolve_profile_name_falls_back_to_env(
    profiles_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KNOMI_PROFILE", "cloud")
    assert resolve_profile_name(None) == "cloud"


def test_resolve_profile_name_falls_back_to_default(profiles_dir: Path) -> None:
    assert resolve_profile_name(None) == "local"


def test_default_profile_is_applied(profiles_dir: Path) -> None:
    config = resolve_config()
    assert config.store.backend == "chroma"
    assert config.store.collection == "kb-local"
    assert config.embedding.dim == 384
    assert config.chunking.strategy == "structure"


def test_explicit_profile_selects_block(profiles_dir: Path) -> None:
    config = resolve_config(profile="cloud")
    assert config.store.backend == "qdrant"
    assert config.store.collection == "kb-cloud"
    assert config.embedding.model == "text-embedding-3-small"


def test_env_overrides_profile(profiles_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOMI_STORE__COLLECTION", "kb-env")
    config = resolve_config(profile="local")
    # env wins over the profile value ("kb-local") ...
    assert config.store.collection == "kb-env"
    # ... but a sibling field from the profile is left untouched.
    assert config.store.backend == "chroma"


def test_overrides_beat_env_and_profile(
    profiles_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KNOMI_STORE__COLLECTION", "kb-env")
    config = resolve_config(profile="local", overrides={"store": {"collection": "kb-cli"}})
    assert config.store.collection == "kb-cli"


def test_override_deep_merges_with_profile(profiles_dir: Path) -> None:
    # Override only chunk_size; the profile's strategy/overlap must survive.
    config = resolve_config(profile="local", overrides={"chunking": {"chunk_size": 1024}})
    assert config.chunking.chunk_size == 1024
    assert config.chunking.strategy == "structure"
    assert config.chunking.chunk_overlap == 32


def test_none_overrides_are_stripped(profiles_dir: Path) -> None:
    # Unset CLI flags arrive as None and must not clobber the profile value.
    config = resolve_config(
        profile="local",
        overrides={"store": {"collection": None}, "chunking": {"strategy": None}},
    )
    assert config.store.collection == "kb-local"
    assert config.chunking.strategy == "structure"


def test_unknown_profile_raises_key_error(profiles_dir: Path) -> None:
    with pytest.raises(KeyError):
        resolve_config(profile="does-not-exist")


def test_secrets_pulled_from_environment(
    profiles_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    config = resolve_config(profile="cloud")  # cloud uses the openai backend
    assert config.embedding.api_key == "sk-from-env"


def test_secrets_not_read_for_mismatched_backend(
    profiles_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # local profile uses the local backend → no API key should be resolved.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    config = resolve_config(profile="local")
    assert config.embedding.api_key is None
