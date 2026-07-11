"""Unit tests for the eval runner (with a fake retriever — no real DB/model)."""

from __future__ import annotations

from math import isclose

import pytest

from knomi.config import Config
from knomi.eval.dataset import EvalQuery
from knomi.eval.runner import evaluate_query, run_eval
from knomi.ingest.chunker import Chunk
from knomi.store.base import SearchResult


def _hit(source: str, score: float) -> SearchResult:
    return SearchResult(chunk=Chunk(text="…", metadata={"source": source}), score=score)


class _FakeRetriever:
    """Returns a scripted result list per successive search() call."""

    def __init__(self, scripted: list[list[SearchResult]]) -> None:
        self._scripted = list(scripted)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return self._scripted.pop(0)[:top_k]


def test_evaluate_query_dedups_docs_and_flags_relevance() -> None:
    hits = [_hit("/x/paper_a.pdf", 0.9), _hit("/x/paper_a.pdf", 0.8), _hit("/x/paper_b.pdf", 0.7)]
    query = EvalQuery(id="q", question="?", relevant_sources=("paper_a.pdf",))
    result = evaluate_query(_FakeRetriever([hits]).search, query, top_k=10)
    assert result.retrieved_docs == ["/x/paper_a.pdf", "/x/paper_b.pdf"]  # deduped
    assert result.ranked_relevance == [True, False]
    assert result.first_relevant_rank == 1


def test_evaluate_query_no_relevant() -> None:
    query = EvalQuery(id="q", question="?", relevant_sources=("target.pdf",))
    result = evaluate_query(_FakeRetriever([[_hit("/x/other.pdf", 0.5)]]).search, query, top_k=10)
    assert result.ranked_relevance == [False]
    assert result.first_relevant_rank is None


def test_run_eval_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    queries = [
        EvalQuery(id="q1", question="one?", relevant_sources=("a.pdf",)),
        EvalQuery(id="q2", question="two?", relevant_sources=("b.pdf",)),
    ]
    scripted = [
        [_hit("/d/a.pdf", 0.9), _hit("/d/c.pdf", 0.8)],  # q1: relevant at rank 1
        [_hit("/d/c.pdf", 0.9), _hit("/d/b.pdf", 0.8)],  # q2: relevant at rank 2
    ]
    monkeypatch.setattr("knomi.retrieval.Retriever", lambda cfg: _FakeRetriever(scripted))

    report = run_eval(Config(), queries, top_k=5)
    assert report.n_queries == 2
    assert report.cutoffs == [1, 3, 5]  # 10 dropped (> top_k), top_k added
    assert isclose(report.metrics["hit@1"], 0.5)
    assert isclose(report.metrics["hit@3"], 1.0)
    assert isclose(report.metrics["mrr"], 0.75)  # mean(1/1, 1/2)


def test_run_eval_default_cutoffs_capped_at_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "knomi.retrieval.Retriever", lambda cfg: _FakeRetriever([[_hit("/d/a.pdf", 1.0)]])
    )
    report = run_eval(
        Config(), [EvalQuery(id="q", question="?", relevant_sources=("a.pdf",))], top_k=3
    )
    assert report.cutoffs == [1, 3]
    assert "recall@3" in report.metrics and "recall@10" not in report.metrics
