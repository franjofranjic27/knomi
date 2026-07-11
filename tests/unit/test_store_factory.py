"""Unit tests for knomi.store.factory.build_store backend routing.

Each backend's client is monkeypatched so no real database (Qdrant server,
Chroma persistence, Postgres) is touched — only the routing logic is verified.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knomi.config import Config
from knomi.store.factory import build_store


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
