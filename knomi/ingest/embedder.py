"""Embedding backend wrapper.

Responsibilities
----------------
- Accept a batch of text strings and return a list of float vectors.
- Support two backends, selected by the ``embedding_model`` config value:
    - **Local** (HuggingFace ``sentence-transformers``): no API key required.
    - **OpenAI API**: ``text-embedding-3-small`` / ``text-embedding-3-large``.
- Respect ``embedding_batch_size`` to avoid OOM / rate-limit errors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from knomi.config import Config
from knomi.ingest.chunker import Chunk


class BaseEmbedder(ABC):
    """Abstract embedding backend."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text string.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, same length and order as *texts*.
        """
        ...

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> list[list[float]]:
        """Embed *chunks* in batches of *batch_size*.

        Args:
            chunks:     Chunks whose ``.text`` will be embedded.
            batch_size: Number of texts to send per backend call.

        Returns:
            Flat list of vectors in the same order as *chunks*.
        """
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors.extend(self.embed([c.text for c in batch]))
        return vectors


class LocalEmbedder(BaseEmbedder):
    """Embedding via a local sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()  # type: ignore[no-any-return]


class OpenAIEmbedder(BaseEmbedder):
    """Embedding via the OpenAI Embeddings API."""

    def __init__(self, model_name: str) -> None:
        import openai

        self.client = openai.OpenAI()
        self.model = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def build_embedder(config: Config) -> BaseEmbedder:
    """Factory — return the correct embedder for *config*.

    If ``embedding_model`` starts with ``text-embedding-``, use OpenAI;
    otherwise treat it as a HuggingFace sentence-transformers model ID.
    """
    if config.embedding_model.startswith("text-embedding-"):
        return OpenAIEmbedder(config.embedding_model)
    return LocalEmbedder(config.embedding_model)
