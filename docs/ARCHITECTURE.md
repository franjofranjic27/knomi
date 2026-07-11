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
                   │ qdrant / chroma │
                   │    / pgvector   │
                   └─────────────────┘
```

Each swappable part — store, embedder, chunker — is chosen at runtime by a factory
that reads the resolved `Config`. Callers depend only on the abstract interface, never
on a concrete backend.

## Module Responsibilities

### `knomi/cli.py`
Typer-based CLI. Defines subcommands (`ingest`, `serve`, `status`, `delete`, `profiles`). A `@app.callback` exposes the global `--profile` / `-p` option (and `--log-level`) and stashes the profile name on the Typer context. Each subcommand collects its flags into a nested overrides dict, drops unset (`None`) values, and calls `resolve_config(profile, overrides)`. No business logic lives here.

### `knomi/config.py`
Pydantic-settings `BaseSettings`. The `Config` model has a few top-level fields (`source_dir`, `serve_host`, `serve_port`, `top_k`) plus three nested groups — `StoreSettings`, `EmbeddingSettings`, `ChunkingSettings`. It is the single source of truth for all tunable parameters.

**Resolution layers** (highest priority first), wired in `settings_customise_sources`:

1. `init_settings` — CLI overrides passed to `resolve_config`.
2. `env_settings` — `KNOMI_`-prefixed environment variables, nested via `__` (e.g. `KNOMI_STORE__COLLECTION`).
3. `dotenv_settings` — the same, read from `.env`.
4. `_ProfileSettingsSource` — the selected profile block from `knomi.json`.

Field defaults sit below all sources.

**Profile resolver.** `resolve_config(profile, overrides)`:
- strips `None`-valued keys from `overrides` so unset flags never clobber lower layers;
- sets the active profile name in a `ContextVar` (`_ACTIVE_PROFILE`) so the profile source knows which block to load — this keeps resolution isolated per thread/async context;
- constructs the `Config` (triggering the four sources above);
- calls `_resolve_secrets()` to fill unset secrets from the environment based on the chosen backend.

`_ProfileSettingsSource` is a custom `PydanticBaseSettingsSource`. On call it resolves the effective profile name (`resolve_profile_name`: explicit arg > `KNOMI_PROFILE` > file `default_profile`), loads `knomi.json` (`load_profiles`, searched in the CWD then `~/.config/knomi/`), and returns the matching profile dict — raising a helpful `KeyError` listing available profiles if the name is unknown.

`_resolve_secrets` reads `OPENAI_API_KEY` / `COHERE_API_KEY` (per embedding backend), `QDRANT_API_KEY` (qdrant store), and `KNOMI_PG_DSN` (pgvector store). Secrets are never persisted in `knomi.json`.

### `knomi/ingest/scanner.py`
Recursive `pathlib` walk. Filters by extension whitelist (`.pdf`, `.md`, `.txt`, `.docx`, `.html`). Returns an iterator of `Path` objects. Computes a SHA-256 hash per file for deduplication.

### `knomi/ingest/parser.py`
Extension-dispatched text extraction. Each format (`pdf`, `md`, `txt`, …) has its own extractor function. Output is normalised plain text: de-hyphenated, whitespace-collapsed, header/footer stripped.

### `knomi/ingest/chunker.py`
Three interchangeable `BaseChunker` strategies, all expressing `chunk_size` / `chunk_overlap` in tokens (via a tiktoken encoding), returning `list[Chunk(text, metadata)]`:
- `TokenChunker` (`token`) — fixed-size sliding window over the raw token stream.
- `StructureChunker` (`structure`) — split on blank lines / Markdown headings, then greedily pack whole blocks up to the token limit.
- `SentenceChunker` (`sentence`) — split into sentences, then greedily pack them up to the token limit.

`build_chunker(config)` selects the strategy from `config.chunking.strategy` via the `_STRATEGIES` registry.

### `knomi/ingest/embedder.py`
`BaseEmbedder` subclasses wrap one embedding backend each and return `list[list[float]]` for a batch of texts. `embed_query` and `embed_chunks` (batching + optional concurrent workers) are inherited. Backends: `LocalEmbedder` (`sentence-transformers`), `OpenAIEmbedder`, `CohereEmbedder`, `OllamaEmbedder`. `build_embedder(config)` routes on `config.embedding.backend`, with a legacy fallback that treats a `text-embedding-*` model as OpenAI and everything else as local.

### `knomi/ingest/pipeline.py`
Orchestrator: scanner → dedup check → parser → chunker → embedder → store. Emits progress via a callback so the CLI can render a progress bar. All steps are pure functions; the pipeline wires them together.

### `knomi/store/base.py`
Abstract `VectorStore` protocol. Methods: `upsert`, `search`, `delete`, `has_document`, `describe`, `get_source`, `update_source`. Any store backend must satisfy this interface.

### `knomi/store/factory.py`
`build_store(config)` returns the concrete `VectorStore` for `config.store.backend` (`qdrant` → `QdrantStore`, `chroma` → `ChromaStore`, `pgvector` → `PgVectorStore`). Backends are imported lazily so optional dependencies (chromadb, psycopg/pgvector) are only required when actually used. The pipeline and serve layers obtain their store exclusively through this factory.

### `knomi/store/qdrant.py`
Qdrant implementation. Uses the official `qdrant-client` SDK. Handles both local-file mode (no Docker) and server mode (HTTP/gRPC URL), plus Qdrant Cloud via `store.url` (https) + `store.api_key`.

### `knomi/serve/server.py`
FastAPI application for the RAG query API. Exposes `GET /health` (liveness check) and `POST /query` (embed a query string, search the vector store, return ranked chunks). The `create_app(config)` factory builds the app; `start_server(config)` runs it with uvicorn.

## Factory pattern

Three factories decouple the orchestration layer from concrete implementations. Each reads
one nested config group and returns an object behind an abstract base class:

| Factory | Reads | Base class | Returns |
|---------|-------|------------|---------|
| `build_store` (`knomi/store/factory.py`) | `config.store.backend` | `VectorStore` | `QdrantStore` / `ChromaStore` / `PgVectorStore` |
| `build_embedder` (`knomi/ingest/embedder.py`) | `config.embedding.backend` | `BaseEmbedder` | `OpenAI` / `Local` / `Cohere` / `Ollama` embedder |
| `build_chunker` (`knomi/ingest/chunker.py`) | `config.chunking.strategy` | `BaseChunker` | `Token` / `Structure` / `Sentence` chunker |

The pipeline (`knomi/ingest/pipeline.py`) and serve layer call these factories and then
program against the base classes only — adding a backend never touches the orchestration code.

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

Nested groups use the `__` delimiter in env vars (e.g. `KNOMI_STORE__COLLECTION`).

| Key | Env var | Default |
|-----|---------|---------|
| `source_dir` | `KNOMI_SOURCE_DIR` | `.` |
| `store.backend` | `KNOMI_STORE__BACKEND` | `qdrant` |
| `store.url` | `KNOMI_STORE__URL` | `http://localhost:6333` |
| `store.path` | `KNOMI_STORE__PATH` | `None` |
| `store.collection` | `KNOMI_STORE__COLLECTION` | `knomi` |
| `embedding.backend` | `KNOMI_EMBEDDING__BACKEND` | `openai` |
| `embedding.model` | `KNOMI_EMBEDDING__MODEL` | `text-embedding-3-small` |
| `embedding.dim` | `KNOMI_EMBEDDING__DIM` | `1536` |
| `chunking.strategy` | `KNOMI_CHUNKING__STRATEGY` | `token` |
| `chunking.chunk_size` | `KNOMI_CHUNKING__CHUNK_SIZE` | `512` |
| `chunking.chunk_overlap` | `KNOMI_CHUNKING__CHUNK_OVERLAP` | `64` |
| `serve_port` | `KNOMI_SERVE_PORT` | `8080` |
| `top_k` | `KNOMI_TOP_K` | `5` |

Secrets are resolved separately from the environment (never from `knomi.json`):
`OPENAI_API_KEY`, `COHERE_API_KEY`, `QDRANT_API_KEY`, `KNOMI_PG_DSN`.

## Extension Points

### Add a vector store backend
1. Create `knomi/store/<backend>.py`, subclass `VectorStore`, implement all abstract methods (`upsert`, `search`, `delete`, `has_document`, `describe`, `get_source`, `update_source`).
2. Add a lazy-import branch for it in `build_store` (`knomi/store/factory.py`) and extend the `store.backend` `Literal` in `knomi/config.py`.
3. If it needs a heavy dependency, add an extra in `pyproject.toml`; import it lazily.
4. Add integration tests under `tests/integration/`.

### Add an embedding backend
1. Subclass `BaseEmbedder` in `knomi/ingest/embedder.py` and implement `embed(texts)` (`embed_query` / `embed_chunks` are inherited).
2. Add a routing branch in `build_embedder` and extend the `embedding.backend` `Literal`.
3. Add unit tests that mock the client and assert the routing.

### Add a chunking strategy
1. Subclass `BaseChunker` in `knomi/ingest/chunker.py`, implement `split(text, source, doc_id)`.
2. Register it in the `_STRATEGIES` dict and extend the `chunking.strategy` `Literal`.
3. Add unit tests in `tests/unit/test_chunker.py`.

## Infrastructure

`compose.yml` runs Qdrant with a named volume for persistence. No other services are required for local development. For production deployments, point `store.url` at a managed Qdrant Cloud instance and provide `QDRANT_API_KEY`.

## Distribution

The canonical artifact is the PyPI package (`pip install knomi`, `requires-python >= 3.12`).
Two thin wrappers install and delegate to it, so there is a single source of truth for the CLI:

- **npm** (`packaging/npm/`) — a `postinstall` hook installs the PyPI package via `uv` / `pipx` / `pip --user` (never failing `npm install`); `bin/knomi.js` shims to the real `knomi` console script.
- **Homebrew** (`Formula/knomi.rb`) — builds knomi and its runtime deps into an isolated virtualenv via `Language::Python::Virtualenv`; `resource` blocks are generated with `brew update-python-resources` and bumped automatically on release.
