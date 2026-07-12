"""Unit tests for knomi.store.factory.build_store backend routing.

Each backend's client is monkeypatched so no real database (Qdrant server,
Chroma persistence, Postgres) is touched — only the routing logic is verified.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knomi.config import Config, StoreSettings
from knomi.store.factory import build_store, store_target


def test_build_store_returns_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    import knomi.store.qdrant as qmod

    # Neutralise the network client and collection bootstrap.
    monkeypatch.setattr(qmod, "QdrantClient", lambda *a, **k: object())
    monkeypatch.setattr(qmod.QdrantStore, "_ensure_collection", lambda self: None)

    config = Config(store={"backend": "qdrant", "url": "http://localhost:6333"})
    store = build_store(config)
    assert isinstance(store, qmod.QdrantStore)


def test_build_store_returns_chroma(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import chromadb

    import knomi.store.chroma as cmod

    monkeypatch.setattr(
        chromadb, "PersistentClient", lambda *a, **k: _FakeChromaClient(), raising=True
    )
    config = Config(store={"backend": "chroma", "path": str(tmp_path)})
    store = build_store(config)
    assert isinstance(store, cmod.ChromaStore)


def test_build_store_returns_pgvector(monkeypatch: pytest.MonkeyPatch) -> None:
    import pgvector.psycopg as pgmod
    import psycopg

    import knomi.store.pgvector as pvmod

    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: _FakePgConn(), raising=True)
    monkeypatch.setattr(pgmod, "register_vector", lambda conn: None, raising=True)

    config = Config(store={"backend": "pgvector", "dsn": "postgresql://localhost/knomi"})
    store = build_store(config)
    assert isinstance(store, pvmod.PgVectorStore)


def test_build_store_pgvector_requires_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    import psycopg

    import knomi.store.pgvector as pvmod  # noqa: F401  (ensures module import)

    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: _FakePgConn(), raising=True)
    config = Config(store={"backend": "pgvector"})
    with pytest.raises(ValueError, match="DSN"):
        build_store(config)


def test_build_store_unknown_backend_raises() -> None:
    config = Config()
    # Bypass the Literal validation to exercise the factory's guard clause.
    object.__setattr__(config.store, "backend", "mystery")
    with pytest.raises(ValueError, match="Unknown store backend"):
        build_store(config)


# --------------------------------------------------------------------------- #
# store_target: secret-safe, backend-aware display of the store location.
# StoreSettings is built directly so the result is independent of any knomi.json
# profile that may sit in the working directory.
# --------------------------------------------------------------------------- #
def test_store_target_chroma_shows_path() -> None:
    store = StoreSettings(backend="chroma", path="./.knomi/chroma")
    assert store_target(store) == "./.knomi/chroma"


def test_store_target_chroma_falls_back_to_url() -> None:
    store = StoreSettings(backend="chroma", url="http://localhost:8000")
    assert store_target(store) == "http://localhost:8000"


def test_store_target_qdrant_shows_url() -> None:
    store = StoreSettings(backend="qdrant", url="https://cluster.qdrant.io:6333")
    assert store_target(store) == "https://cluster.qdrant.io:6333"


def test_store_target_pgvector_uri_omits_password() -> None:
    store = StoreSettings(
        backend="pgvector", dsn="postgresql://user:secret@db.example.com:5432/knomi"
    )
    target = store_target(store)
    assert target == "db.example.com:5432/knomi"
    assert "secret" not in target


def test_store_target_pgvector_keyword_form_omits_password() -> None:
    store = StoreSettings(
        backend="pgvector",
        dsn="host=db.example.com port=5432 dbname=knomi user=u password=secret",
    )
    target = store_target(store)
    assert target == "db.example.com:5432/knomi"
    assert "secret" not in target


def test_store_target_pgvector_without_dsn() -> None:
    store = StoreSettings(backend="pgvector")
    assert store_target(store) == "pgvector"


# --------------------------------------------------------------------------- #
# Minimal fakes for backend clients (construction side effects only).
# --------------------------------------------------------------------------- #
class _FakeChromaCollection:
    def count(self) -> int:
        return 0


class _FakeChromaClient:
    def get_or_create_collection(self, *args: object, **kwargs: object) -> _FakeChromaCollection:
        return _FakeChromaCollection()


class _FakePgConn:
    autocommit = True

    def execute(self, *args: object, **kwargs: object) -> None:
        return None
