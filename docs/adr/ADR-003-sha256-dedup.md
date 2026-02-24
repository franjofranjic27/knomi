# ADR-003: SHA-256 for document deduplication

**Status:** Accepted
**Date:** 2025-01

## Context

Re-embedding unchanged files wastes API credits and compute. knomi needs a way to detect whether a file has already been ingested and can be skipped.

## Options considered

| Strategy | Detects content change | Path-independent | Cost |
|----------|----------------------|-----------------|------|
| **SHA-256 of file content** | ✓ | ✓ | ~10 ms/MB |
| File mtime | ✗ (can be touched) | ✗ | O(1) |
| File size | ✗ (same size ≠ same content) | ✗ | O(1) |
| MD5 | ✓ | ✓ | slightly faster, collision risk |
| Relative file path | ✗ | ✗ | O(1) |

## Decision

**SHA-256 of file content**, streamed in 64 KiB chunks (`knomi/ingest/scanner.py`).

Key reasons:
1. **Content-addressed**: the same file under any path or name produces the same hash — dedup works across renames and moves.
2. **Collision-resistant**: SHA-256 has no known practical collisions; MD5 was ruled out due to known vulnerabilities.
3. **Path-independent**: enables move detection — if a hash already exists in the store but under a different source path, the pipeline updates the metadata without re-embedding.

## Stored as `doc_id`

Each `Chunk.metadata["doc_id"]` holds the SHA-256 of the originating file. This value is:
- Stored in Qdrant payload for every chunk.
- Used as the dedup key in `has_document(doc_id)`.
- Used as the deletion key in `delete(doc_id)`.
- Combined with `chunk_index` to generate a deterministic UUID5 point ID, making upserts idempotent.

## Move detection

When `has_document(sha256)` returns `True` but `get_source(sha256)` returns a different path than the current scan path, the pipeline calls `update_source(sha256, new_path)` to update the Qdrant payload — no re-embedding required.

## Consequences

- Two files with identical content are treated as one document (same `doc_id`). This is intentional and desirable.
- Files that change content get a new hash and are re-embedded on the next ingest run. The old vectors remain until `knomi delete` is implemented.
- Hashing adds ~10 ms per MB of file content. For typical document sizes (< 10 MB), this is negligible.
