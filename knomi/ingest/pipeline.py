"""Ingestion pipeline orchestrator.

Flow
----
scan → dedup → parse → chunk → embed → store

Each step is a pure function or thin wrapper; this module wires them together
and handles progress reporting and error recovery.

Not yet implemented — stubs only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from knomi.config import Config


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run."""

    total_files: int = 0
    skipped_files: int = 0  # already indexed (hash match)
    failed_files: list[Path] = field(default_factory=list)
    total_chunks: int = 0
    total_vectors: int = 0


def run_pipeline(
    config: Config,
    progress_callback: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Execute the full ingest pipeline for *config.source_dir*.

    Args:
        config:            Runtime configuration (paths, model, DB, etc.).
        progress_callback: Optional function called with a status string at
                           each major step (used by the CLI to render progress).

    Returns:
        ``PipelineResult`` with counts of processed / skipped / failed files.

    TODO — step-by-step implementation plan:

    1. **Scan**
       ``from knomi.ingest.scanner import scan``
       Iterate ``scan(config.source_dir)`` to get ``ScannedFile`` objects.

    2. **Dedup**
       ``from knomi.store.qdrant import QdrantStore``
       Instantiate the store. For each ``ScannedFile``, check whether a
       document with the same ``sha256`` already exists in the collection.
       Increment ``result.skipped_files`` and skip if so.

    3. **Parse**
       ``from knomi.ingest.parser import parse``
       Call ``parse(scanned_file.path)`` → plain text string.
       On empty result or exception: add to ``result.failed_files``, continue.

    4. **Chunk**
       ``from knomi.ingest.chunker import chunk``
       Call ``chunk(text, source=..., doc_id=..., chunk_size=..., chunk_overlap=...)``
       → ``list[Chunk]``.

    5. **Embed**
       ``from knomi.ingest.embedder import build_embedder``
       ``embedder = build_embedder(config)``
       Call ``embedder.embed_chunks(chunks, batch_size=config.embedding_batch_size)``
       → ``list[list[float]]``.

    6. **Store**
       Call ``store.upsert(chunks, vectors)`` to persist into Qdrant.
       Increment ``result.total_vectors``.

    7. **Return** ``PipelineResult``.
    """
    raise NotImplementedError
