"""ChromaDB vector store implementation.

A zero-infrastructure backend: runs fully in-process with on-disk persistence
(``config.store.path``) or against a Chroma server (``config.store.url``).
Cosine distance is used to match the other backends.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, cast

from knomi.config import Config
from knomi.ingest.chunker import Chunk
from knomi.store.base import SearchResult, VectorStore

log = logging.getLogger(__name__)


def _point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic id so re-indexing the same content is idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id + str(chunk_index)))


class ChromaStore(VectorStore):
    """ChromaDB-backed vector store."""

    def __init__(self, config: Config) -> None:
        import chromadb

        store = config.store
        self.collection_name = store.collection
        # An explicit on-disk path means local mode. Only fall back to a Chroma
        # server when no path is given and the url is an HTTP(S) endpoint.
        if store.path:
            client = chromadb.PersistentClient(path=store.path)
        elif store.url.startswith(("http://", "https://")):
            from urllib.parse import urlparse

            parsed = urlparse(store.url)
            client = chromadb.HttpClient(
                host=parsed.hostname or "localhost", port=parsed.port or 8000
            )
        else:
            client = chromadb.PersistentClient(path=store.url)
        # Cosine space keeps scores comparable with Qdrant/pgvector.
        self.collection = client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[_point_id(c.metadata["doc_id"], c.metadata["chunk_index"]) for c in chunks],
            embeddings=cast(Any, vectors),
            documents=[c.text for c in chunks],
            metadatas=[dict(c.metadata) for c in chunks],
        )

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        res = self.collection.query(
            query_embeddings=cast(Any, [vector]),
            n_results=top_k,
            include=cast(Any, ["documents", "metadatas", "distances"]),
        )
        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        return [
            SearchResult(
                chunk=Chunk(text=doc, metadata=dict(meta or {})),
                # Cosine distance → similarity score in [-1, 1].
                score=1.0 - float(dist),
            )
            for doc, meta, dist in zip(documents, metadatas, distances, strict=True)
        ]

    def delete(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def has_document(self, doc_id: str) -> bool:
        got = self.collection.get(where={"doc_id": doc_id}, limit=1)
        return bool(got.get("ids"))

    def get_source(self, doc_id: str) -> str | None:
        got = self.collection.get(where={"doc_id": doc_id}, limit=1, include=["metadatas"])
        metadatas = got.get("metadatas") or []
        if not metadatas:
            return None
        return str((metadatas[0] or {}).get("source", "")) or None

    def update_source(self, doc_id: str, new_source: str) -> None:
        got = self.collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        ids = got.get("ids") or []
        metadatas = got.get("metadatas") or []
        if not ids:
            return
        self.collection.update(
            ids=ids,
            metadatas=[{**(m or {}), "source": new_source} for m in metadatas],
        )

    def describe(self) -> dict[str, object]:
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "points_count": count,
            "indexed_vectors_count": count,
        }
