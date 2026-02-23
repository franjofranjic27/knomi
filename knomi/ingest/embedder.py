"""Embedding backend wrapper.

Responsibilities
----------------
- Accept a batch of text strings and return a list of float vectors.
- Support two backends, selected by the ``embedding_model`` config value:
    - **Local** (HuggingFace ``sentence-transformers``): no API key required.
    - **OpenAI API**: ``text-embedding-3-small`` / ``text-embedding-3-large``.
- Respect ``embedding_batch_size`` to avoid OOM / rate-limit errors.

Not yet implemented — stubs only.
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

        TODO:
            - Slice chunks into batches.
            - Call ``self.embed()`` for each batch.
            - Flatten and return results in original order.
        """
        raise NotImplementedError


class LocalEmbedder(BaseEmbedder):
    """Embedding via a local sentence-transformers model.

    TODO:
        - Load ``SentenceTransformer(model_name)`` in ``__init__``.
        - Implement ``embed()`` using ``model.encode(texts).tolist()``.
    """

    def __init__(self, model_name: str) -> None:
        # TODO: self.model = SentenceTransformer(model_name)
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbedder(BaseEmbedder):
    """Embedding via the OpenAI Embeddings API.

    TODO:
        - Initialise ``openai.OpenAI()`` client in ``__init__``.
        - Implement ``embed()`` using ``client.embeddings.create()``.
        - Handle rate-limit retries with exponential back-off.
    """

    def __init__(self, model_name: str) -> None:
        # TODO: self.client = openai.OpenAI(); self.model = model_name
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


def build_embedder(config: Config) -> BaseEmbedder:
    """Factory — return the correct embedder for *config*.

    Heuristic: if ``embedding_model`` starts with ``text-embedding-``,
    use the OpenAI backend; otherwise treat it as a HuggingFace model ID.

    TODO: Implement routing logic.
    """
    raise NotImplementedError
