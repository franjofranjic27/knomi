"""Pure ranking-quality metrics for retrieval evaluation.

All functions operate on a ranked binary relevance list — ``relevances[i]`` is
``True`` if the document at rank ``i`` (0-based, most similar first) is relevant
to the query. Keeping these pure and dependency-free makes them trivial to test
and reason about.
"""

from __future__ import annotations

from math import log2


def hit_rate_at_k(relevances: list[bool], k: int) -> float:
    """1.0 if at least one relevant document appears in the top *k*, else 0.0.

    Also known as Success@k — answers "did we surface *anything* useful?".
    """
    return 1.0 if any(relevances[:k]) else 0.0


def recall_at_k(relevances: list[bool], k: int, total_relevant: int) -> float:
    """Fraction of all relevant documents that appear in the top *k*.

    Args:
        relevances:     Ranked binary relevance flags for retrieved documents.
        k:              Cutoff rank.
        total_relevant: Number of documents that *should* be retrieved for the
                        query (the recall denominator). ``0`` yields ``0.0``.
    """
    if total_relevant <= 0:
        return 0.0
    return sum(relevances[:k]) / total_relevant


def precision_at_k(relevances: list[bool], k: int) -> float:
    """Fraction of the top *k* retrieved documents that are relevant."""
    if k <= 0:
        return 0.0
    window = relevances[:k]
    if not window:
        return 0.0
    return sum(window) / len(window)


def reciprocal_rank(relevances: list[bool], k: int | None = None) -> float:
    """Reciprocal of the rank of the first relevant document (0.0 if none).

    Averaged over queries this is the Mean Reciprocal Rank (MRR).
    """
    window = relevances if k is None else relevances[:k]
    for index, is_relevant in enumerate(window):
        if is_relevant:
            return 1.0 / (index + 1)
    return 0.0


def dcg_at_k(relevances: list[bool], k: int) -> float:
    """Discounted cumulative gain with binary gains at cutoff *k*."""
    return sum(1.0 / log2(i + 2) for i, rel in enumerate(relevances[:k]) if rel)


def ndcg_at_k(relevances: list[bool], k: int, total_relevant: int) -> float:
    """Normalised DCG at *k* — DCG divided by the ideal DCG.

    The ideal ranking places ``min(total_relevant, k)`` relevant documents first.
    Returns ``0.0`` when there is no attainable gain.
    """
    ideal_hits = min(total_relevant, k)
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
    if idcg <= 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg
