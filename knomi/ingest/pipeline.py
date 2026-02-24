"""Ingestion pipeline orchestrator.

Flow
----
scan → dedup → parse → chunk → embed → store

Each step is a pure function or thin wrapper; this module wires them together
and handles progress reporting, move detection, and error recovery.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from knomi.config import Config

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run."""

    total_files: int = 0
    skipped_files: int = 0  # already indexed (hash match, same path)
    moved_files: int = 0  # already indexed but source path changed
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
        ``PipelineResult`` with counts of processed / skipped / moved / failed files.
    """
    from knomi.ingest.chunker import chunk
    from knomi.ingest.embedder import build_embedder
    from knomi.ingest.parser import parse
    from knomi.ingest.scanner import scan
    from knomi.store.qdrant import QdrantStore

    result = PipelineResult()
    store = QdrantStore(config)
    embedder = build_embedder(config)

    log.info("Starting ingest pipeline for %s", config.source_dir)

    for scanned in scan(config.source_dir):
        result.total_files += 1
        log.debug("Processing %s", scanned.path.name)
        if progress_callback:
            progress_callback(f"Processing {scanned.path.name}")

        if store.has_document(scanned.sha256):
            stored_source = store.get_source(scanned.sha256)
            if stored_source and stored_source != str(scanned.path):
                log.info(
                    "File moved: %s → %s (updating metadata)",
                    stored_source,
                    scanned.path,
                )
                store.update_source(scanned.sha256, str(scanned.path))
                result.moved_files += 1
            else:
                log.debug("Skipping %s (already indexed)", scanned.path.name)
            result.skipped_files += 1
            continue

        try:
            text = parse(scanned.path)
        except Exception:
            log.warning("Failed to parse %s", scanned.path, exc_info=True)
            result.failed_files.append(scanned.path)
            continue

        if not text:
            log.warning("Empty text extracted from %s", scanned.path)
            result.failed_files.append(scanned.path)
            continue

        chunks = chunk(
            text,
            source=scanned.path,
            doc_id=scanned.sha256,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        vectors = embedder.embed_chunks(
            chunks,
            batch_size=config.embedding_batch_size,
            workers=config.embedding_workers,
        )
        store.upsert(chunks, vectors)
        result.total_chunks += len(chunks)
        result.total_vectors += len(vectors)
        log.debug("Indexed %s: %d chunks, %d vectors", scanned.path.name, len(chunks), len(vectors))

    log.info(
        "Pipeline complete: %d files, %d skipped, %d moved, %d failed, %d vectors",
        result.total_files,
        result.skipped_files,
        result.moved_files,
        len(result.failed_files),
        result.total_vectors,
    )
    return result
