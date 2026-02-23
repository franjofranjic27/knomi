"""Qdrant vector store implementation.

Supports two modes:
- **Server mode**: ``db_url`` is an HTTP/gRPC URL (e.g. ``http://localhost:6333``).
- **Local mode**: ``db_url`` is a filesystem path (e.g. ``./qdrant_local``);
  Qdrant runs in-process with no Docker required.

Not yet implemented — stubs only.
"""

from __future__ import annotations

from knomi.config import Config
from knomi.ingest.chunker import Chunk
from knomi.store.base import SearchResult, VectorStore


class QdrantStore(VectorStore):
    """Qdrant-backed vector store.

    TODO — implementation plan:

    ``__init__``:
        - Import ``qdrant_client.QdrantClient``.
        - If ``config.db_url`` starts with ``http://`` or ``grpc://``, create
          a remote client: ``QdrantClient(url=config.db_url)``.
        - Otherwise treat it as a local path:
          ``QdrantClient(path=config.db_url)``.
        - Call ``_ensure_collection()`` to create the collection if absent.

    ``_ensure_collection``:
        - Check ``client.collection_exists(config.collection)``.
        - If not, call ``client.create_collection(...)`` with
          ``VectorParams(size=config.embedding_dim, distance=Distance.COSINE)``.

    ``upsert``:
        - Build ``PointStruct`` objects: id = UUID derived from
          ``chunk.metadata["doc_id"] + str(chunk_index)``, vector = vector,
          payload = chunk.metadata + {"text": chunk.text}.
        - Call ``client.upsert(collection_name=..., points=[...])``.

    ``search``:
        - Call ``client.search(collection_name=..., query_vector=..., limit=top_k)``.
        - Map results to ``SearchResult(chunk=Chunk(...), score=hit.score)``.

    ``delete``:
        - Use a filter on ``payload["doc_id"]`` to delete all matching points.

    ``has_document``:
        - ``client.count(collection_name=..., count_filter=Filter(...))``.
        - Return ``count > 0``.

    ``describe``:
        - Return ``client.get_collection(config.collection)`` info as a dict.
    """

    def __init__(self, config: Config) -> None:
        # TODO: initialise qdrant_client
        raise NotImplementedError

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

    def delete(self, doc_id: str) -> None:
        raise NotImplementedError

    def has_document(self, doc_id: str) -> bool:
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        raise NotImplementedError
