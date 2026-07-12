"""Unit tests for the pure retrieval metrics."""

from __future__ import annotations

from math import isclose, log2

import pytest

from knomi.eval import metrics


def test_hit_rate() -> None:
    assert metrics.hit_rate_at_k([False, True, False], 3) == 1.0
    assert metrics.hit_rate_at_k([False, True, False], 1) == 0.0
    assert metrics.hit_rate_at_k([], 5) == 0.0


def test_recall() -> None:
    # 2 of 3 relevant docs appear in the top 5.
    assert isclose(metrics.recall_at_k([True, False, True, False], 5, total_relevant=3), 2 / 3)
    # Cutoff limits what counts.
    assert isclose(metrics.recall_at_k([True, False, True], 1, total_relevant=2), 1 / 2)
    assert metrics.recall_at_k([True], 5, total_relevant=0) == 0.0


def test_precision() -> None:
    assert isclose(metrics.precision_at_k([True, False, True, False], 4), 0.5)
    assert metrics.precision_at_k([], 5) == 0.0
    assert metrics.precision_at_k([True], 0) == 0.0


def test_reciprocal_rank() -> None:
    assert metrics.reciprocal_rank([False, False, True]) == 1 / 3
    assert metrics.reciprocal_rank([True, True]) == 1.0
    assert metrics.reciprocal_rank([False, False, False]) == 0.0
    # A relevant doc beyond the cutoff does not count.
    assert metrics.reciprocal_rank([False, False, True], k=2) == 0.0


def test_dcg_matches_formula() -> None:
    # Relevant at ranks 1 and 3 (0-based 0 and 2).
    expected = 1 / log2(2) + 1 / log2(4)
    assert isclose(metrics.dcg_at_k([True, False, True], 3), expected)


def test_ndcg_perfect_and_partial() -> None:
    # All relevant docs ranked first -> nDCG 1.0.
    assert isclose(metrics.ndcg_at_k([True, True, False], 3, total_relevant=2), 1.0)
    # Relevant doc pushed to rank 2 -> below ideal.
    ndcg = metrics.ndcg_at_k([False, True], 2, total_relevant=1)
    assert 0.0 < ndcg < 1.0
    assert isclose(ndcg, (1 / log2(3)) / (1 / log2(2)))


def test_ndcg_no_relevant() -> None:
    assert metrics.ndcg_at_k([False, False], 2, total_relevant=0) == 0.0
    assert metrics.ndcg_at_k([False, False], 2, total_relevant=2) == 0.0


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_metrics_bounded_between_zero_and_one(k: int) -> None:
    rel = [True, False, False, True, False, False, True, False, False, False]
    assert 0.0 <= metrics.hit_rate_at_k(rel, k) <= 1.0
    assert 0.0 <= metrics.recall_at_k(rel, k, total_relevant=3) <= 1.0
    assert 0.0 <= metrics.ndcg_at_k(rel, k, total_relevant=3) <= 1.0
    assert 0.0 <= metrics.reciprocal_rank(rel, k) <= 1.0
