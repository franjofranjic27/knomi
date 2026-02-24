"""Qdrant vector store implementation.

Supports two modes:
- **Server mode**: ``db_url`` is an HTTP/gRPC URL (e.g. ``http://localhost:6333``).
- **Local mode**: ``db_url`` is a filesystem path (e.g. ``./qdrant_local``);
  Qdrant runs in-process with no Docker required.
"""

from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from knomi.config import Config
from knomi.ingest.chunker import Chunk
from knomi.store.base import SearchResult, VectorStore

log = logging.getLogger(__name__)


class QdrantStore(VectorStore):
    """Qdrant-backed vector store."""

    def __init__(self, config: Config) -> None:
        self.collection = config.collection
        self.dim = config.embedding_dim
        if config.db_url.startswith(("http://", "grpc://")):
            self.client = QdrantClient(url=config.db_url)
        else:
            self.client = QdrantClient(path=config.db_url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    @retry(
        wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        # uuid5(NAMESPACE_DNS, doc_id + chunk_index) produces a deterministic ID
        # so re-indexing the same content is idempotent (upsert, not insert).
        points = [
            PointStruct(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        c.metadata["doc_id"] + str(c.metadata["chunk_index"]),
                    )
                ),
                vector=v,
                payload={**c.metadata, "text": c.text},
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        hits = self.client.query_points(
            collection_name=self.collection, query=vector, limit=top_k
        ).points
        return [
            SearchResult(
                chunk=Chunk(
                    text=(h.payload or {})["text"],
                    metadata={k: v for k, v in (h.payload or {}).items() if k != "text"},
                ),
                score=h.score,
            )
            for h in hits
        ]

    def delete(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )

    def has_document(self, doc_id: str) -> bool:
        return (
            self.client.count(
                collection_name=self.collection,
                count_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            ).count
            > 0
        )

    def get_source(self, doc_id: str) -> str | None:
        """Return the ``source`` path stored for the first chunk of *doc_id*."""
        records, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=1,
            with_payload=["source"],
        )
        if not records:
            return None
        return str((records[0].payload or {}).get("source", ""))

    def update_source(self, doc_id: str, new_source: str) -> None:
        """Overwrite the ``source`` field for every chunk belonging to *doc_id*."""
        self.client.set_payload(
            collection_name=self.collection,
            payload={"source": new_source},
            points=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )

    def describe(self) -> dict[str, object]:
        info = self.client.get_collection(self.collection)
        return {
            "name": self.collection,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
        }
