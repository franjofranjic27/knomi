"""Query-time retrieval pipeline.

Composes the three query-time components — embedder, vector store, and optional
reranker — into one object so every caller (serve, eval, future clients) shares
identical retrieval semantics instead of re-wiring the steps:

    embed query  →  store.search(top_n)  →  rerank → top_k

When reranking is disabled the store is queried for exactly ``top_k`` results and
returned as-is.
"""

from __future__ import annotations

from knomi.config import Config
from knomi.store.base import SearchResult


class Retriever:
    """Embeds queries and retrieves (optionally reranked) chunks for a config."""

    def __init__(self, config: Config) -> None:
        from knomi.ingest.embedder import build_embedder
        from knomi.rerank import build_reranker
        from knomi.store.factory import build_store

        self.config = config
        self.embedder = build_embedder(config)
        self.store = build_store(config)
        self.reranker = build_reranker(config)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return the top *top_k* chunks for *query*.

        With a reranker, ``reranking.top_n`` candidates are fetched first (never
        fewer than *top_k*) and re-scored down to *top_k*; without one, the store
        is queried for *top_k* directly.
        """
        vector = self.embedder.embed_query(query)
        if self.reranker is None:
            return self.store.search(vector, top_k=top_k)
        pool = max(self.config.reranking.top_n, top_k)
        candidates = self.store.search(vector, top_k=pool)
        return self.reranker.rerank(query, candidates, top_k=top_k)
