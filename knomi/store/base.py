"""Abstract VectorStore interface.

All store backends (Qdrant, ChromaDB, …) must satisfy this protocol.
The pipeline and serve layers depend only on this interface — never on
a concrete implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from knomi.ingest.chunker import Chunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single result from a similarity search."""

    chunk: Chunk
    score: float


class VectorStore(ABC):
    """Abstract base for all vector store backends."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update *chunks* and their corresponding *vectors*.

        Each chunk's ``metadata["doc_id"]`` is used as the dedup key.
        """
        ...

    @abstractmethod
    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return the *top_k* most similar chunks to *vector*."""
        ...

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Remove all vectors associated with *doc_id* from the collection."""
        ...

    @abstractmethod
    def has_document(self, doc_id: str) -> bool:
        """Return True if any vector with *doc_id* exists in the collection."""
        ...

    @abstractmethod
    def describe(self) -> dict[str, object]:
        """Return collection metadata (vector count, dimension, etc.)."""
        ...

    @abstractmethod
    def get_source(self, doc_id: str) -> str | None:
        """Return the stored source path for *doc_id*, or ``None`` if not indexed."""
        ...

    @abstractmethod
    def update_source(self, doc_id: str, new_source: str) -> None:
        """Update the ``source`` field for every chunk belonging to *doc_id*."""
        ...
