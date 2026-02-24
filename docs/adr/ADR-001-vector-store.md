# ADR-001: Qdrant as the default vector store

**Status:** Accepted
**Date:** 2025-01

## Context

knomi needs a vector store to persist embeddings and serve similarity searches. Requirements:
- Works locally without a cloud account (developer-friendly)
- Runs in Docker for team setups
- Scales to a managed cloud offering without code changes
- Supports filtering by metadata fields (e.g. `doc_id` for dedup and deletion)
- Active development and good Python SDK

## Options considered

| Option | Local | Cloud | Filtering | Notes |
|--------|-------|-------|-----------|-------|
| **Qdrant** | ✓ in-process or Docker | ✓ Qdrant Cloud | ✓ | gRPC + REST, active OSS project |
| ChromaDB | ✓ in-process | limited | partial | Simpler API, less scalable |
| Milvus | Docker only | ✓ Zilliz | ✓ | Heavy, Kubernetes-native |
| Pinecone | ✗ cloud only | ✓ | ✓ | No local mode |
| Weaviate | Docker | ✓ | ✓ | More complex setup |

## Decision

**Qdrant** is the default backend.

Key reasons:
1. **In-process mode** (`QdrantClient(path=...)`) works without Docker — zero friction for first-time users.
2. **Same code, different URL** to switch from local → Qdrant Cloud — no interface changes.
3. **Payload filtering** supports efficient dedup (`has_document`), move detection (`get_source`), and targeted deletion (`delete`).
4. **UUID5-based point IDs** make upserts idempotent — re-indexing the same content is safe.

## Consequences

- ChromaDB is documented as a potential alternative but not implemented (see `VectorStore` interface in `knomi/store/base.py`).
- Any new backend must implement all seven abstract methods: `upsert`, `search`, `delete`, `has_document`, `describe`, `get_source`, `update_source`.
- Qdrant server mode requires Docker (or Qdrant Cloud) — acceptable for production use.
