"""Unit tests for the three chunking strategies and build_chunker factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from knomi.config import ChunkingSettings, Config
from knomi.ingest.chunker import (
    BaseChunker,
    SentenceChunker,
    StructureChunker,
    TokenChunker,
    build_chunker,
)

_SOURCE = Path("docs/sample.md")
_DOC_ID = "sha256-deadbeef"

# A multi-paragraph, multi-sentence Markdown text so structure/sentence
# strategies have real boundaries to split on.
_TEXT = (
    "# Heading One\n\n"
    "The first paragraph has two sentences. Here is the second one.\n\n"
    "## Heading Two\n\n"
    "A third paragraph follows. It also has several sentences! Does it work?"
)


def _settings(strategy: str, **overrides: object) -> ChunkingSettings:
    base: dict[str, object] = {"strategy": strategy, "chunk_size": 512, "chunk_overlap": 64}
    base.update(overrides)
    return ChunkingSettings(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("strategy", "chunker_cls"),
    [
        ("token", TokenChunker),
        ("structure", StructureChunker),
        ("sentence", SentenceChunker),
    ],
)
def test_strategy_produces_nonempty_chunks_with_metadata(
    strategy: str, chunker_cls: type[BaseChunker]
) -> None:
    chunker = chunker_cls(_settings(strategy))
    chunks = chunker.split(_TEXT, source=_SOURCE, doc_id=_DOC_ID)

    assert chunks, "expected at least one chunk"
    assert all(c.text.strip() for c in chunks)
    total = len(chunks)
    for index, c in enumerate(chunks):
        assert c.metadata["source"] == str(_SOURCE)
        assert c.metadata["doc_id"] == _DOC_ID
        assert c.metadata["chunk_index"] == index
        assert c.metadata["total_chunks"] == total


@pytest.mark.parametrize("strategy", ["token", "structure", "sentence"])
def test_empty_text_returns_no_chunks(strategy: str) -> None:
    chunker = build_chunker(Config(chunking={"strategy": strategy}))
    assert chunker.split("", source=_SOURCE, doc_id=_DOC_ID) == []


def test_structure_windows_oversize_block() -> None:
    # A single block far larger than the token limit must be split, not dropped.
    big_block = " ".join(["alpha"] * 200)
    chunker = StructureChunker(_settings("structure", chunk_size=20, chunk_overlap=4))
    chunks = chunker.split(big_block, source=_SOURCE, doc_id=_DOC_ID)
    assert len(chunks) > 1
    assert chunks[-1].metadata["total_chunks"] == len(chunks)


def test_sentence_windows_oversize_sentence() -> None:
    # One sentence longer than the limit is windowed on its own.
    long_sentence = " ".join(["beta"] * 200) + "."
    chunker = SentenceChunker(_settings("sentence", chunk_size=20, chunk_overlap=4))
    chunks = chunker.split(long_sentence, source=_SOURCE, doc_id=_DOC_ID)
    assert len(chunks) > 1


def test_token_windows_oversize_text() -> None:
    words = " ".join(["gamma"] * 100)
    chunker = TokenChunker(_settings("token", chunk_size=10, chunk_overlap=2))
    chunks = chunker.split(words, source=_SOURCE, doc_id=_DOC_ID)
    assert len(chunks) > 1
    assert chunks[0].metadata["chunk_index"] == 0


@pytest.mark.parametrize(
    ("strategy", "expected_cls"),
    [
        ("token", TokenChunker),
        ("structure", StructureChunker),
        ("sentence", SentenceChunker),
    ],
)
def test_build_chunker_selects_class(strategy: str, expected_cls: type[BaseChunker]) -> None:
    chunker = build_chunker(Config(chunking={"strategy": strategy}))
    assert isinstance(chunker, expected_cls)
    assert chunker.settings.strategy == strategy


@pytest.mark.parametrize("strategy", ["token", "structure", "sentence"])
def test_chunker_handles_literal_special_tokens(strategy: str) -> None:
    """Documents containing special-token strings (e.g. '<|endoftext|>', common
    in ML/NLP literature) must be encoded as plain text, not raise a ValueError.
    """
    text = (
        "Tokenizers reserve markers. The GPT vocabulary uses <|endoftext|> to "
        "separate documents. Some models also add <|endofprompt|> tokens.\n\n"
        "A second paragraph repeats the token <|endoftext|> once more."
    )
    # Force windowing so the encode path is exercised on the special-token text.
    chunker = build_chunker(Config(chunking={"strategy": strategy, "chunk_size": 12}))
    chunks = chunker.split(text, source=_SOURCE, doc_id=_DOC_ID)
    assert chunks
    assert "<|endoftext|>" in " ".join(c.text for c in chunks)
