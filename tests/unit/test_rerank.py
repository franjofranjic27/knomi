"""Unit tests for reranking backends and build_reranker routing.

Backend clients (sentence-transformers CrossEncoder, Cohere) are monkeypatched
so no model is downloaded and no network is touched.
"""

from __future__ import annotations

import pytest

from knomi.config import Config
from knomi.ingest.chunker import Chunk
from knomi.rerank import CohereReranker, LocalReranker, build_reranker
from knomi.store.base import SearchResult


def _results(*texts: str) -> list[SearchResult]:
    return [
        SearchResult(chunk=Chunk(text=t, metadata={"source": f"{t}.pdf"}), score=0.0) for t in texts
    ]


class _FakeCrossEncoder:
    """predict() returns the length of each candidate text as its score."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(doc)) for _query, doc in pairs]


def test_build_reranker_disabled_returns_none() -> None:
    assert build_reranker(Config()) is None
    assert build_reranker(Config(reranking={"enabled": False})) is None


def test_build_reranker_local(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _FakeCrossEncoder, raising=True)
    reranker = build_reranker(Config(reranking={"enabled": True, "backend": "local"}))
    assert isinstance(reranker, LocalReranker)


def test_build_reranker_cohere(monkeypatch: pytest.MonkeyPatch) -> None:
    import cohere

    monkeypatch.setattr(cohere, "Client", lambda api_key=None: object(), raising=True)
    reranker = build_reranker(
        Config(reranking={"enabled": True, "backend": "cohere", "api_key": "x"})
    )
    assert isinstance(reranker, CohereReranker)


def test_local_reranker_orders_by_score_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _FakeCrossEncoder, raising=True)
    reranker = LocalReranker("fake-model")
    # Scores = text length: "long text" > "mid" > "x".
    ranked = reranker.rerank("q", _results("x", "long text", "mid"), top_k=2)
    assert [r.chunk.text for r in ranked] == ["long text", "mid"]
    assert ranked[0].score == float(len("long text"))


def test_local_reranker_empty() -> None:
    # No CrossEncoder call needed for the empty short-circuit.
    reranker = LocalReranker.__new__(LocalReranker)  # type: ignore[call-arg]
    assert reranker.rerank("q", [], top_k=5) == []


def test_cohere_reranker_maps_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    import cohere

    class _Item:
        def __init__(self, index: int, score: float) -> None:
            self.index = index
            self.relevance_score = score

    class _Resp:
        # Cohere returns items ordered best-first, referencing input indices.
        results = [_Item(2, 0.9), _Item(0, 0.4)]

    class _Client:
        def __init__(self, api_key: str | None = None) -> None: ...
        def rerank(self, **kwargs: object) -> _Resp:
            return _Resp()

    monkeypatch.setattr(cohere, "Client", _Client, raising=True)
    reranker = CohereReranker("rerank-v3", api_key="x")
    ranked = reranker.rerank("q", _results("a", "b", "c"), top_k=2)
    assert [r.chunk.text for r in ranked] == ["c", "a"]  # by input index 2 then 0
    assert ranked[0].score == 0.9
