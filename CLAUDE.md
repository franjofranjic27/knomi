# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**knomi** is an open-source CLI tool and background service with two distinct modes:

1. **Ingest mode** — Recursively scans a folder for PDFs (and similar documents), runs them through a token-efficient embedding pipeline, and stores vectors in a local or remote vector DB.
2. **Serve/connect mode** — Connects a running vector DB as a RAG source for AI agents (Claude, OpenWebUI, Ollama, etc.).

The entire embedding pipeline is Python. The CLI entry point wraps the pipeline and exposes configuration flags. A `compose.yml` brings up all required services (vector DB, optional embedding worker) with a single command.

## Architecture

```
knomi/
├── compose.yml               # Qdrant only — port 6333 (REST) + 6334 (gRPC)
├── pyproject.toml            # uv + hatchling; `knomi` script entry point
├── knomi/
│   ├── __main__.py           # `python -m knomi` entry point
│   ├── cli.py                # Typer CLI — subcommands: ingest, serve, status
│   ├── config.py             # Pydantic BaseSettings — all tunable params
│   ├── ingest/
│   │   ├── scanner.py        # Recursive walk, extension filter, SHA-256 hash
│   │   ├── parser.py         # Extension-dispatched text extraction (PDF, md, txt)
│   │   ├── chunker.py        # tiktoken sliding-window splitter → list[Chunk]
│   │   ├── embedder.py       # BaseEmbedder, LocalEmbedder, OpenAIEmbedder, factory
│   │   └── pipeline.py       # Orchestrator: scan→dedup→parse→chunk→embed→store
│   ├── store/
│   │   ├── base.py           # Abstract VectorStore (upsert, search, delete, has_document, describe, get_source, update_source)
│   │   └── qdrant.py         # Qdrant implementation (server URL or local file path)
│   └── serve/
│       ├── __init__.py
│       └── server.py         # FastAPI app — GET /health, POST /query
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TESTING.md
│   ├── COMMITING_CONVENTION.md
│   └── WORKFLOWS.md
├── .github/
│   ├── CODEOWNERS
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       ├── ci.yml            # lint → unit tests → integration tests (with Qdrant service)
│       └── release.yml       # PyPI publish on v*.*.* tags via OIDC
└── tests/
    ├── conftest.py           # tmp_docs_dir, mock_embedder, qdrant_url fixtures
    ├── unit/
    └── integration/
```

## Key Design Decisions

### Token Efficiency
- Parse PDFs to plain text first (strip headers/footers, de-hyphenate, normalise whitespace) before any tokenisation.
- Chunk with a token counter (e.g. `tiktoken` or the embedding model's tokeniser), not character count, so `chunk_size` is always in tokens.
- Avoid re-embedding unchanged documents: store a SHA-256 hash of each source file and skip if already indexed.

### Configuration Hierarchy
CLI flags > environment variables > `knomi.toml` / `.env` file. Pydantic `BaseSettings` handles merging. Key options:
- `--source-dir` / `SOURCE_DIR`
- `--chunk-size` (tokens, default 512)
- `--chunk-overlap` (tokens, default 64)
- `--embedding-model` (e.g. `text-embedding-3-small`, `nomic-embed-text`)
- `--db-url` (e.g. `http://localhost:6333` for Qdrant, or local path for ChromaDB)
- `--collection` (vector DB collection name)

### Vector DB
Default target is **Qdrant** (runs well locally via Docker and scales to cloud). ChromaDB is the lightweight fallback for zero-infrastructure setups. Both implement the same `VectorStore` abstract interface so they are interchangeable.

### Compose Services
`compose.yml` should define at minimum:
- `qdrant` service on port 6333 with a named volume for persistence.
- `knomi-worker` service that runs `python -m knomi ingest` on startup (optional, can be disabled).

## Development Setup

```bash
# Install Python deps (assuming uv)
uv sync

# Run CLI directly
python -m knomi --help

# Start infrastructure only
docker compose up qdrant -d

# Full stack (DB + ingest worker)
docker compose up
```

## Running Tests

```bash
pytest                        # all tests
pytest tests/unit/            # unit tests only (no Docker required)
pytest tests/integration/     # requires compose services running
pytest -k "test_chunker"      # single test by name
```

## CLI Subcommands (target UX)

```bash
knomi ingest ./docs --chunk-size 512 --chunk-overlap 64
knomi ingest ./docs --db-url http://qdrant:6333 --collection my-kb
knomi status                  # show indexed collections + doc count
knomi serve --port 8080       # expose RAG as HTTP API for agents
```

## Supported File Types

Priority order for the scanner: `.pdf`, `.md`, `.txt`, `.docx`, `.html`. Each format has its own parser; the pipeline routes by extension.

## How to Extend

### Adding a new document format

1. Add the extension to `SUPPORTED_EXTENSIONS` in `knomi/ingest/scanner.py`.
2. Write a `_parse_<format>(path: Path) -> str` function in `knomi/ingest/parser.py`.
   - Must return normalised plain text or `""` on failure — never raise.
   - Call `_normalise()` before returning.
   - Lazy-import the parsing library inside the function body.
3. Register it in the `extractors` dict in `parse()` (`knomi/ingest/parser.py`).
4. Add a unit test in `tests/unit/test_parser.py`: create a minimal fixture, call `parse()`, assert the expected text is present.

### Adding a new vector store backend

1. Create `knomi/store/<backend>.py`.
2. Subclass `VectorStore` from `knomi/store/base.py` and implement **all** abstract methods:
   `upsert`, `search`, `delete`, `has_document`, `describe`, `get_source`, `update_source`.
3. Wire it into `knomi/ingest/pipeline.py` and `knomi/serve/server.py` via a factory
   (similar to `build_embedder()` in `knomi/ingest/embedder.py`).
4. Add integration tests under `tests/integration/`.

### Adding a new embedding backend

1. Subclass `BaseEmbedder` in `knomi/ingest/embedder.py` and implement `embed(texts)`.
   The `embed_query` and `embed_chunks` (including concurrent batching) are inherited for free.
2. Update the `build_embedder()` factory with the routing condition.
3. Add unit tests in `tests/unit/test_embedder.py` — mock the backend, test the routing logic.

### Key data flow (ingest)

```
ScannedFile(path, sha256)
  → parse(path)          → plain text   (knomi/ingest/parser.py)
  → chunk(text, ...)     → list[Chunk]  (knomi/ingest/chunker.py)
  → embedder.embed(...)  → list[vector] (knomi/ingest/embedder.py)
  → store.upsert(...)                   (knomi/store/qdrant.py)
```

Each `Chunk.metadata` carries `doc_id` (SHA-256), `source` (path), `chunk_index`, and `total_chunks`.
These fields are stored as Qdrant payload and are used for dedup, move detection, and deletion.

## Commit Style

Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`.
