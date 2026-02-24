"""Embedding backend wrapper.

Responsibilities
----------------
- Accept a batch of text strings and return a list of float vectors.
- Support two backends, selected by the ``embedding_model`` config value:
    - **Local** (HuggingFace ``sentence-transformers``): no API key required.
    - **OpenAI API**: ``text-embedding-3-small`` / ``text-embedding-3-large``.
- Respect ``embedding_batch_size`` to avoid OOM / rate-limit errors.
- Optionally embed batches concurrently via ``embed_chunks(workers=N)``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import openai as _openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from knomi.config import Config
from knomi.ingest.chunker import Chunk

log = logging.getLogger(__name__)


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

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string and return its vector.

        Args:
            text: The query string to embed.

        Returns:
            A single float vector.
        """
        return self.embed([text])[0]

    def embed_chunks(
        self, chunks: list[Chunk], batch_size: int = 64, workers: int = 1
    ) -> list[list[float]]:
        """Embed *chunks* in batches, optionally using concurrent workers.

        Args:
            chunks:     Chunks whose ``.text`` will be embedded.
            batch_size: Number of texts to send per backend call.
            workers:    Number of concurrent threads for parallel batch dispatch.
                        Values > 1 are most beneficial for API-backed embedders
                        (e.g. OpenAI) where network latency dominates.

        Returns:
            Flat list of vectors in the same order as *chunks*.
        """
        if not chunks:
            return []

        batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

        def _embed_batch(batch: list[Chunk]) -> list[list[float]]:
            return self.embed([c.text for c in batch])

        if workers > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_embed_batch, batches))
        else:
            results = [_embed_batch(b) for b in batches]

        return [v for batch in results for v in batch]


class LocalEmbedder(BaseEmbedder):
    """Embedding via a local sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()  # type: ignore[no-any-return]


class OpenAIEmbedder(BaseEmbedder):
    """Embedding via the OpenAI Embeddings API."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self.client = _openai.OpenAI(api_key=api_key)
        self.model = model_name

    @retry(
        retry=retry_if_exception_type(
            (_openai.RateLimitError, _openai.APIConnectionError, _openai.APITimeoutError)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def build_embedder(config: Config) -> BaseEmbedder:
    """Factory — return the correct embedder for *config*.

    If ``embedding_model`` starts with ``text-embedding-``, use OpenAI;
    otherwise treat it as a HuggingFace sentence-transformers model ID.
    """
    if config.embedding_model.startswith("text-embedding-"):
        return OpenAIEmbedder(config.embedding_model, api_key=config.openai_api_key)
    return LocalEmbedder(config.embedding_model)
