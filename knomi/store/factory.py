"""Vector store factory.

Returns the concrete :class:`~knomi.store.base.VectorStore` implementation for
``config.store.backend``. The pipeline and serve layers depend only on the
abstract interface and obtain their store through this factory — mirroring
:func:`knomi.ingest.embedder.build_embedder`.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from knomi.config import Config, StoreSettings
from knomi.store.base import VectorStore


def _pgvector_target(dsn: str | None) -> str:
    """Describe a Postgres DSN as ``host[:port][/dbname]`` — never the password.

    Handles both the URI form (``postgresql://user:pass@host:5432/db``) and the
    keyword form (``host=... port=... dbname=... password=...``). Falls back to
    the bare backend name if nothing parseable (and safe) can be extracted.
    """
    if not dsn:
        return "pgvector"
    parsed = urlparse(dsn)
    if parsed.hostname:
        uri_port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.hostname}{uri_port}{parsed.path}"
    # Keyword form: pull only the non-sensitive fields, never `password`.
    host = re.search(r"\bhost=(\S+)", dsn)
    port = re.search(r"\bport=(\S+)", dsn)
    dbname = re.search(r"\bdbname=(\S+)", dsn)
    if host:
        suffix = f":{port.group(1)}" if port else ""
        db = f"/{dbname.group(1)}" if dbname else ""
        return f"{host.group(1)}{suffix}{db}"
    return "pgvector"


def store_target(store: StoreSettings) -> str:
    """Human-readable, secret-safe description of where the store lives.

    Used by CLI output so each backend reports its real target instead of an
    unrelated default (e.g. a local Chroma path rather than the Qdrant URL).
    """
    if store.backend == "chroma":
        return store.path or store.url
    if store.backend == "pgvector":
        return _pgvector_target(store.dsn)
    # qdrant: the api_key is a separate field and never part of the URL.
    return store.url


def build_store(config: Config) -> VectorStore:
    """Return the vector store backend selected by ``config.store.backend``.

    Backends are imported lazily so that optional dependencies (chromadb,
    psycopg/pgvector) are only required when the corresponding backend is used.
    """
    backend = config.store.backend
    if backend == "qdrant":
        from knomi.store.qdrant import QdrantStore

        return QdrantStore(config)
    if backend == "chroma":
        from knomi.store.chroma import ChromaStore

        return ChromaStore(config)
    if backend == "pgvector":
        from knomi.store.pgvector import PgVectorStore

        return PgVectorStore(config)
    raise ValueError(f"Unknown store backend: {backend!r}")  # pragma: no cover
