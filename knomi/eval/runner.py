"""Run a gold set against a configured, already-indexed collection.

For each question the runner embeds the query, retrieves the top-k chunks, maps
them to their unique source documents (keeping each document's best rank), and
scores the ranking with the metrics in :mod:`knomi.eval.metrics`. Results are
aggregated (mean over queries) at several rank cutoffs.

The runner only *reads* the store — it never ingests — so evaluating a profile
is fast and non-destructive.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from knomi.config import Config
from knomi.eval import metrics
from knomi.eval.dataset import EvalQuery, source_matches

_DEFAULT_CUTOFFS = (1, 3, 5, 10)


@dataclass(slots=True)
class QueryResult:
    """Per-query evaluation outcome."""

    id: str
    question: str
    total_relevant: int
    ranked_relevance: list[bool]  # one flag per retrieved unique document, by rank
    retrieved_docs: list[str]

    @property
    def first_relevant_rank(self) -> int | None:
        """1-based rank of the first relevant document, or ``None`` if missed."""
        for index, rel in enumerate(self.ranked_relevance):
            if rel:
                return index + 1
        return None


@dataclass(slots=True)
class EvalReport:
    """Aggregate evaluation report over all queries."""

    top_k: int
    cutoffs: list[int]
    n_queries: int
    metrics: dict[str, float]  # flat: "recall@5", "ndcg@10", "hit@1", "mrr", …
    per_query: list[QueryResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable summary (metrics + light per-query detail)."""
        return {
            "top_k": self.top_k,
            "cutoffs": self.cutoffs,
            "n_queries": self.n_queries,
            "metrics": self.metrics,
            "per_query": [
                {
                    "id": q.id,
                    "question": q.question,
                    "total_relevant": q.total_relevant,
                    "first_relevant_rank": q.first_relevant_rank,
                    "retrieved_docs": q.retrieved_docs,
                }
                for q in self.per_query
            ],
        }


def _rank_documents(sources: list[str]) -> list[str]:
    """Collapse a ranked chunk-source list to unique documents, best rank first."""
    seen: set[str] = set()
    ranked: list[str] = []
    for source in sources:
        if source and source not in seen:
            seen.add(source)
            ranked.append(source)
    return ranked


def _relevance_flags(ranked_docs: list[str], relevant_sources: tuple[str, ...]) -> list[bool]:
    return [any(source_matches(doc, rel) for rel in relevant_sources) for doc in ranked_docs]


def evaluate_query(
    store: object,
    embedder: object,
    query: EvalQuery,
    top_k: int,
) -> QueryResult:
    """Retrieve for a single *query* and return its ranked relevance."""
    vector = embedder.embed_query(query.question)  # type: ignore[attr-defined]
    hits = store.search(vector, top_k=top_k)  # type: ignore[attr-defined]
    ranked_docs = _rank_documents([h.chunk.metadata.get("source", "") for h in hits])
    return QueryResult(
        id=query.id,
        question=query.question,
        total_relevant=len(query.relevant_sources),
        ranked_relevance=_relevance_flags(ranked_docs, query.relevant_sources),
        retrieved_docs=ranked_docs,
    )


def _aggregate(results: list[QueryResult], top_k: int, cutoffs: list[int]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {}
    out: dict[str, float] = {}
    for k in cutoffs:
        out[f"hit@{k}"] = sum(metrics.hit_rate_at_k(r.ranked_relevance, k) for r in results) / n
        out[f"recall@{k}"] = (
            sum(metrics.recall_at_k(r.ranked_relevance, k, r.total_relevant) for r in results) / n
        )
        out[f"ndcg@{k}"] = (
            sum(metrics.ndcg_at_k(r.ranked_relevance, k, r.total_relevant) for r in results) / n
        )
    out["mrr"] = sum(metrics.reciprocal_rank(r.ranked_relevance, top_k) for r in results) / n
    return out


def run_eval(
    config: Config,
    queries: list[EvalQuery],
    top_k: int = 10,
    cutoffs: list[int] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> EvalReport:
    """Evaluate *queries* against the collection described by *config*.

    Args:
        config:            Resolved configuration (selects store + embedder).
        queries:           Gold questions to run.
        top_k:             Number of chunks retrieved per query.
        cutoffs:           Rank cutoffs to report; defaults to ``1,3,5,10``
                           filtered to ``<= top_k`` (``top_k`` always included).
        progress_callback: Optional per-query progress hook.

    Returns:
        An :class:`EvalReport` with aggregate metrics and per-query detail.
    """
    from knomi.ingest.embedder import build_embedder
    from knomi.store.factory import build_store

    if cutoffs is None:
        cutoffs = sorted({c for c in _DEFAULT_CUTOFFS if c <= top_k} | {top_k})

    store = build_store(config)
    embedder = build_embedder(config)

    results: list[QueryResult] = []
    for query in queries:
        if progress_callback:
            progress_callback(query.question)
        results.append(evaluate_query(store, embedder, query, top_k))

    return EvalReport(
        top_k=top_k,
        cutoffs=cutoffs,
        n_queries=len(results),
        metrics=_aggregate(results, top_k, cutoffs),
        per_query=results,
    )
