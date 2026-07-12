"""Unit tests for the Retriever query pipeline (embed → search → rerank)."""

from __future__ import annotations

import pytest

from knomi.config import Config
from knomi.ingest.chunker import Chunk
from knomi.retrieval import Retriever
from knomi.store.base import SearchResult


def _results(n: int) -> list[SearchResult]:
    return [
        SearchResult(chunk=Chunk(text=f"c{i}", metadata={"source": f"{i}"}), score=1.0 / (i + 1))
        for i in range(n)
    ]


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class _FakeStore:
    def __init__(self) -> None:
        self.last_top_k: int | None = None

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        self.last_top_k = top_k
        return _results(top_k)


class _FakeReranker:
    def __init__(self) -> None:
        self.last_top_k: int | None = None
        self.last_n_candidates: int | None = None

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        self.last_top_k = top_k
        self.last_n_candidates = len(results)
        return results[:top_k]


def _patch(monkeypatch: pytest.MonkeyPatch, store: object, reranker: object) -> None:
    monkeypatch.setattr("knomi.ingest.embedder.build_embedder", lambda cfg: _FakeEmbedder())
    monkeypatch.setattr("knomi.store.factory.build_store", lambda cfg: store)
    monkeypatch.setattr("knomi.rerank.build_reranker", lambda cfg: reranker)


def test_retriever_without_reranker_queries_store_for_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeStore()
    _patch(monkeypatch, store, None)
    retriever = Retriever(Config())
    hits = retriever.search("q", top_k=4)
    assert store.last_top_k == 4  # no candidate pool without a reranker
    assert len(hits) == 4


def test_retriever_with_reranker_fetches_pool_then_reranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, reranker = _FakeStore(), _FakeReranker()
    _patch(monkeypatch, store, reranker)
    config = Config(reranking={"enabled": True, "top_n": 20})
    retriever = Retriever(config)
    hits = retriever.search("q", top_k=3)
    assert store.last_top_k == 20  # candidate pool = top_n
    assert reranker.last_n_candidates == 20
    assert reranker.last_top_k == 3
    assert len(hits) == 3


def test_retriever_pool_never_smaller_than_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    store, reranker = _FakeStore(), _FakeReranker()
    _patch(monkeypatch, store, reranker)
    config = Config(reranking={"enabled": True, "top_n": 5})
    retriever = Retriever(config)
    retriever.search("q", top_k=8)  # top_k > top_n
    assert store.last_top_k == 8
