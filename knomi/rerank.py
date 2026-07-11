"""Cross-encoder reranking — an optional second retrieval stage.

Bi-encoder vector search (the store) is fast but scores query and document
independently. A cross-encoder re-scores each ``(query, chunk)`` pair jointly,
which is markedly more accurate for relevance. The typical pipeline is:

    retrieve top-N candidates (cheap)  →  rerank to top-K (accurate)

Two backends, selected by ``config.reranking.backend``:
- **local**  — a HuggingFace ``CrossEncoder`` (no API key; ships with
  ``sentence-transformers``).
- **cohere** — the Cohere Rerank API.

``build_reranker`` returns ``None`` when reranking is disabled, so callers can
treat "no reranker" as a first-class, allocation-free case.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from knomi.config import Config
from knomi.store.base import SearchResult

log = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract reranking backend."""

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Re-score *results* for *query* and return the best *top_k*.

        The returned results carry the reranker's relevance score (not the
        original vector similarity) and are ordered best-first.
        """
        ...


class LocalReranker(BaseReranker):
    """Reranking via a local HuggingFace cross-encoder."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        scores = self.model.predict([(query, r.chunk.text) for r in results])
        ranked = sorted(zip(results, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [SearchResult(chunk=r.chunk, score=float(s)) for r, s in ranked[:top_k]]


class CohereReranker(BaseReranker):
    """Reranking via the Cohere Rerank API."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        import cohere

        self.client = cohere.Client(api_key=api_key)
        self.model = model_name

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        response = self.client.rerank(
            query=query,
            documents=[r.chunk.text for r in results],
            model=self.model,
            top_n=min(top_k, len(results)),
        )
        return [
            SearchResult(chunk=results[item.index].chunk, score=float(item.relevance_score))
            for item in response.results
        ]


def build_reranker(config: Config) -> BaseReranker | None:
    """Return the reranker for *config*, or ``None`` when reranking is disabled."""
    settings = config.reranking
    if not settings.enabled:
        return None
    if settings.backend == "local":
        return LocalReranker(settings.model)
    if settings.backend == "cohere":
        return CohereReranker(settings.model, api_key=settings.api_key)
    raise ValueError(f"Unknown reranker backend: {settings.backend!r}")  # pragma: no cover
