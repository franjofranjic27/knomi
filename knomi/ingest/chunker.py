"""Token-aware text chunkers.

Three interchangeable strategies split normalised plain text into overlapping
:class:`Chunk` windows. All strategies express ``chunk_size`` / ``chunk_overlap``
in **tokens** (via a tiktoken encoding), never characters, so limits match the
embedding model's budget:

- ``token``     — fixed-size sliding window over the raw token stream.
- ``structure`` — split on blank lines / Markdown headings, then greedily pack
                  whole blocks up to the token limit (keeps paragraphs intact).
- ``sentence``  — split into sentences, then greedily pack sentences up to the
                  token limit (keeps sentences intact).

The concrete strategy is selected by :func:`build_chunker` from
``config.chunking.strategy``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from knomi.config import ChunkingSettings, Config


@dataclass(slots=True)
class Chunk:
    """A text window ready for embedding.

    Attributes:
        text:      The chunk's text content.
        metadata:  Arbitrary key/value pairs stored alongside the vector
                   (source path, chunk index, doc SHA-256, …).
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_chunks(texts: list[str], source: Path, doc_id: str) -> list[Chunk]:
    """Wrap ordered chunk *texts* into ``Chunk`` objects with shared metadata."""
    total = len(texts)
    return [
        Chunk(
            text=t,
            metadata={
                "source": str(source),
                "doc_id": doc_id,
                "chunk_index": idx,
                "total_chunks": total,
            },
        )
        for idx, t in enumerate(texts)
    ]


def chunk(
    text: str,
    source: Path,
    doc_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split *text* into overlapping token windows (the ``token`` strategy).

    Retained as a module-level function for direct use and backward
    compatibility; :class:`TokenChunker` delegates to it.

    Args:
        text:           Normalised plain text from the parser.
        source:         Original file path (stored in chunk metadata).
        doc_id:         SHA-256 of the source file (for dedup / deletion).
        chunk_size:     Maximum number of tokens per chunk.
        chunk_overlap:  Number of tokens shared between adjacent chunks.
        encoding_name:  tiktoken encoding name; must match the embedding model.

    Returns:
        Ordered list of ``Chunk`` objects.
    """
    if not text:
        return []

    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    # disallowed_special=() so documents that literally contain special-token
    # strings (e.g. "<|endoftext|>", common in ML/NLP literature) are encoded as
    # plain text instead of raising a ValueError.
    tokens = enc.encode(text, disallowed_special=())

    if len(tokens) <= chunk_size:
        return _make_chunks([text], source, doc_id)

    # Sliding-window algorithm:
    #   step  = chunk_size - chunk_overlap   (how far to advance each iteration)
    #   Each window spans tokens[i : i + chunk_size].
    #   Adjacent windows share `chunk_overlap` tokens, giving the embedding
    #   model context continuity across chunk boundaries.
    step = max(1, chunk_size - chunk_overlap)
    windows: list[list[int]] = []
    i = 0
    while i < len(tokens):
        windows.append(tokens[i : i + chunk_size])
        i += step

    return _make_chunks([enc.decode(w) for w in windows], source, doc_id)


def _pack_units(
    units: list[str],
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str,
) -> list[str]:
    """Greedily pack text *units* into chunks not exceeding *chunk_size* tokens.

    A unit longer than *chunk_size* on its own is token-windowed so no content is
    dropped. Consecutive chunks share up to *chunk_overlap* tokens for context.

    Args:
        units:          Ordered text fragments to pack (paragraphs or sentences).
        chunk_size:     Maximum tokens per chunk.
        chunk_overlap:  Approximate token overlap carried between chunks.
        encoding_name:  tiktoken encoding name.

    Returns:
        Ordered list of chunk texts.
    """
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append(" ".join(current).strip())
            current = []
            current_tokens = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        n = len(enc.encode(unit, disallowed_special=()))

        # A single oversized unit: window it on its own after flushing.
        if n > chunk_size:
            flush()
            tokens = enc.encode(unit, disallowed_special=())
            step = max(1, chunk_size - chunk_overlap)
            for i in range(0, len(tokens), step):
                chunks.append(enc.decode(tokens[i : i + chunk_size]).strip())
            continue

        if current_tokens + n > chunk_size:
            flush()
            # Carry overlap: re-seed with the trailing chunk_overlap tokens of the
            # chunk we just flushed, so context continues across the boundary.
            if chunk_overlap > 0 and chunks:
                carry_text = enc.decode(
                    enc.encode(chunks[-1], disallowed_special=())[-chunk_overlap:]
                ).strip()
                if carry_text:
                    current = [carry_text]
                    current_tokens = len(enc.encode(carry_text, disallowed_special=()))

        current.append(unit)
        current_tokens += n

    flush()
    return [c for c in chunks if c]


class BaseChunker(ABC):
    """Abstract chunking strategy."""

    def __init__(self, settings: ChunkingSettings) -> None:
        self.settings = settings

    @abstractmethod
    def split(self, text: str, source: Path, doc_id: str) -> list[Chunk]:
        """Split *text* into ``Chunk`` objects. Returns ``[]`` for empty text."""
        ...


class TokenChunker(BaseChunker):
    """Fixed-size sliding window over the raw token stream."""

    def split(self, text: str, source: Path, doc_id: str) -> list[Chunk]:
        return chunk(
            text,
            source=source,
            doc_id=doc_id,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            encoding_name=self.settings.encoding_name,
        )


class StructureChunker(BaseChunker):
    """Split on blank lines / Markdown headings, then pack blocks to the limit."""

    # A blank line, or a line starting a Markdown heading, starts a new block.
    _BLOCK_SPLIT = re.compile(r"\n\s*\n+|(?=^#{1,6}\s)", re.MULTILINE)

    def split(self, text: str, source: Path, doc_id: str) -> list[Chunk]:
        if not text:
            return []
        blocks = [b for b in self._BLOCK_SPLIT.split(text) if b and b.strip()]
        texts = _pack_units(
            blocks,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            self.settings.encoding_name,
        )
        return _make_chunks(texts, source, doc_id)


class SentenceChunker(BaseChunker):
    """Split into sentences, then pack sentences to the token limit."""

    # Split on whitespace that follows a sentence-ending punctuation mark.
    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def split(self, text: str, source: Path, doc_id: str) -> list[Chunk]:
        if not text:
            return []
        sentences = [s for s in self._SENTENCE_SPLIT.split(text) if s and s.strip()]
        texts = _pack_units(
            sentences,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            self.settings.encoding_name,
        )
        return _make_chunks(texts, source, doc_id)


_STRATEGIES: dict[str, type[BaseChunker]] = {
    "token": TokenChunker,
    "structure": StructureChunker,
    "sentence": SentenceChunker,
}


def build_chunker(config: Config) -> BaseChunker:
    """Factory — return the chunker for ``config.chunking.strategy``."""
    strategy = config.chunking.strategy
    try:
        return _STRATEGIES[strategy](config.chunking)
    except KeyError:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"Unknown chunking strategy: {strategy!r}") from None
