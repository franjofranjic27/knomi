"""Ingestion pipeline orchestrator.

Flow
----
scan → dedup → parse → chunk → embed → store

Each step is a pure function or thin wrapper; this module wires them together
and handles progress reporting and error recovery.
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
    """
    from knomi.ingest.chunker import chunk
    from knomi.ingest.embedder import build_embedder
    from knomi.ingest.parser import parse
    from knomi.ingest.scanner import scan
    from knomi.store.qdrant import QdrantStore

    result = PipelineResult()
    store = QdrantStore(config)
    embedder = build_embedder(config)

    for scanned in scan(config.source_dir):
        result.total_files += 1
        if progress_callback:
            progress_callback(f"Processing {scanned.path.name}")

        if store.has_document(scanned.sha256):
            result.skipped_files += 1
            continue

        try:
            text = parse(scanned.path)
        except Exception:
            result.failed_files.append(scanned.path)
            continue

        if not text:
            result.failed_files.append(scanned.path)
            continue

        chunks = chunk(
            text,
            source=scanned.path,
            doc_id=scanned.sha256,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        vectors = embedder.embed_chunks(chunks, batch_size=config.embedding_batch_size)
        store.upsert(chunks, vectors)
        result.total_chunks += len(chunks)
        result.total_vectors += len(vectors)

    return result
