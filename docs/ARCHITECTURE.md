# Architecture

## Overview

knomi has two runtime modes that share core infrastructure:

```
┌─────────────────────────────────────────────────────────────────┐
│                          knomi CLI                              │
│                                                                 │
│   knomi ingest <dir>            knomi serve                     │
│         │                             │                         │
│         ▼                             ▼                         │
│   Ingest Pipeline              RAG Query Interface              │
│   ─────────────────            ────────────────────             │
│   Scanner                      VectorStore.search()             │
│     → Parser                   → return ranked chunks           │
│       → Chunker                                                  │
│         → Embedder                                               │
│           → VectorStore.upsert()                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Vector DB     │
                   │   (Qdrant)      │
                   │   local / cloud │
                   └─────────────────┘
```

## Module Responsibilities

### `knomi/cli.py`
Typer-based CLI. Defines subcommands (`ingest`, `serve`, `status`). Parses flags, constructs a `Config` object, and delegates to the relevant service. No business logic lives here.

### `knomi/config.py`
Pydantic `BaseSettings`. Merges CLI flags > environment variables > `.env` file. Single source of truth for all tunable parameters (chunk size, overlap, DB URL, embedding model, etc.).

### `knomi/ingest/scanner.py`
Recursive `pathlib` walk. Filters by extension whitelist (`.pdf`, `.md`, `.txt`, `.docx`, `.html`). Returns an iterator of `Path` objects. Computes a SHA-256 hash per file for deduplication.

### `knomi/ingest/parser.py`
Extension-dispatched text extraction. Each format (`pdf`, `md`, `txt`, …) has its own extractor function. Output is normalised plain text: de-hyphenated, whitespace-collapsed, header/footer stripped.

### `knomi/ingest/chunker.py`
Splits plain text into overlapping token windows using a tokeniser consistent with the chosen embedding model. Respects `chunk_size` and `chunk_overlap` from `Config`. Returns a list of `Chunk(text, metadata)`.

### `knomi/ingest/embedder.py`
Thin wrapper around an embedding backend. Supports local models (via `sentence-transformers`) and API models (OpenAI, etc.). Returns `list[list[float]]` for a batch of texts.

### `knomi/ingest/pipeline.py`
Orchestrator: scanner → dedup check → parser → chunker → embedder → store. Emits progress via a callback so the CLI can render a progress bar. All steps are pure functions; the pipeline wires them together.

### `knomi/store/base.py`
Abstract `VectorStore` protocol. Methods: `upsert(chunks, vectors)`, `search(vector, top_k)`, `delete(doc_id)`, `list_collections()`. Any store backend must satisfy this interface.

### `knomi/store/qdrant.py`
Qdrant implementation. Uses the official `qdrant-client` SDK. Handles both local-file mode (no Docker) and server mode (HTTP/gRPC URL).

## Data Flow: Ingest

```
File path
  │  scanner.py – hash check
  ▼
Raw bytes
  │  parser.py – format-specific extraction
  ▼
Plain text
  │  chunker.py – token-aware splitting
  ▼
[Chunk(text, metadata), ...]
  │  embedder.py – batch embed
  ▼
[vector, ...]
  │  store.upsert()
  ▼
Qdrant collection
```

## Data Flow: Query (serve)

```
User query string
  │  embedder.embed_query()
  ▼
Query vector
  │  store.search(vector, top_k)
  ▼
Ranked [Chunk, score]
  │  returned to agent / HTTP response
  ▼
LLM context
```

## Configuration Contract

| Key | Env var | Default |
|-----|---------|---------|
| `source_dir` | `KNOMI_SOURCE_DIR` | `.` |
| `chunk_size` | `KNOMI_CHUNK_SIZE` | `512` |
| `chunk_overlap` | `KNOMI_CHUNK_OVERLAP` | `64` |
| `embedding_model` | `KNOMI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `db_url` | `KNOMI_DB_URL` | `http://localhost:6333` |
| `collection` | `KNOMI_COLLECTION` | `knomi` |

## Infrastructure

`compose.yml` runs Qdrant with a named volume for persistence. No other services are required for local development. For production deployments, point `KNOMI_DB_URL` at a managed Qdrant Cloud instance.
