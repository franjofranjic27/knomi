"""Token-aware text chunker.

Responsibilities
----------------
- Split normalised plain text into overlapping token windows.
- ``chunk_size`` and ``chunk_overlap`` are always expressed in tokens,
  not characters, using a tokeniser matched to the embedding model.
- Return ``Chunk`` objects that carry the text and source metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Chunk:
    """A text window ready for embedding.

    Attributes:
        text:      The chunk's text content.
        metadata:  Arbitrary key/value pairs stored alongside the vector
                   (source path, page number, chunk index, doc SHA-256, …).
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk(
    text: str,
    source: Path,
    doc_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split *text* into overlapping token windows.

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
    tokens = enc.encode(text)

    if len(tokens) <= chunk_size:
        return [
            Chunk(
                text=text,
                metadata={
                    "source": str(source),
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
            )
        ]

    step = chunk_size - chunk_overlap
    windows: list[list[int]] = []
    i = 0
    while i < len(tokens):
        windows.append(tokens[i : i + chunk_size])
        i += step

    total = len(windows)
    return [
        Chunk(
            text=enc.decode(w),
            metadata={
                "source": str(source),
                "doc_id": doc_id,
                "chunk_index": idx,
                "total_chunks": total,
            },
        )
        for idx, w in enumerate(windows)
    ]
