"""Vector store factory.

Returns the concrete :class:`~knomi.store.base.VectorStore` implementation for
``config.store.backend``. The pipeline and serve layers depend only on the
abstract interface and obtain their store through this factory — mirroring
:func:`knomi.ingest.embedder.build_embedder`.
"""

from __future__ import annotations

from knomi.config import Config
from knomi.store.base import VectorStore


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
