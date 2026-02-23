# knomi

> Token-efficient document ingestion and RAG connector for local AI agents.

[![CI](https://github.com/franjofranjic27/knomi/actions/workflows/ci.yml/badge.svg)](https://github.com/franjofranjic27/knomi/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/knomi)](https://pypi.org/project/knomi/)
[![Python](https://img.shields.io/pypi/pyversions/knomi)](https://pypi.org/project/knomi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- **Recursive document ingestion** — scans a folder for PDFs, Markdown, plain text, DOCX, and HTML files.
- **Token-efficient chunking** — splits text using `tiktoken` so `chunk_size` is always in tokens, not characters.
- **Deduplication** — stores a SHA-256 hash per source file and skips re-embedding unchanged documents.
- **Pluggable embeddings** — local models via `sentence-transformers` or OpenAI-compatible APIs.
- **Pluggable vector stores** — Qdrant (default) or ChromaDB; both share the same abstract interface.
- **RAG serve mode** — exposes the indexed vector store as an HTTP API for Claude, OpenWebUI, Ollama, and other agents.
- **Single-command infrastructure** — `docker compose up` brings up Qdrant and an optional ingest worker.

---

## Quick start

### Option A — Docker (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/franjofranjic27/knomi.git
cd knomi

# 2. Start Qdrant
docker compose up qdrant -d

# 3. Install knomi
pip install knomi          # or: uv add knomi

# 4. Ingest your documents
knomi ingest ./docs --db-url http://localhost:6333 --collection my-kb
```

### Option B — pip install only (ChromaDB, zero infrastructure)

```bash
pip install knomi
knomi ingest ./docs --db-url ./chroma_data --collection my-kb
```

---

## CLI usage

### `ingest` — index documents into the vector store

```bash
# Ingest a folder with default settings
knomi ingest ./docs

# Custom chunk size, overlap, and collection
knomi ingest ./docs --chunk-size 512 --chunk-overlap 64 --collection my-kb

# Remote Qdrant instance
knomi ingest ./docs --db-url http://qdrant:6333 --collection my-kb

# Use OpenAI embeddings
knomi ingest ./docs --embedding-model text-embedding-3-small
```

### `status` — inspect indexed collections

```bash
knomi status
# Shows all collections and their document counts.
```

### `serve` — expose RAG as an HTTP API for agents

```bash
knomi serve --port 8080
# Starts an HTTP server that agents (Claude, OpenWebUI, Ollama) can query.
```

---

## Configuration

All options can be set as CLI flags, environment variables, or in a `knomi.toml` / `.env` file.
Precedence: **CLI flags > env vars > config file**.

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `--source-dir` | `SOURCE_DIR` | `.` | Folder to scan for documents |
| `--chunk-size` | `CHUNK_SIZE` | `512` | Chunk size in tokens |
| `--chunk-overlap` | `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks (tokens) |
| `--embedding-model` | `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `--db-url` | `KNOMI_DB_URL` | `http://localhost:6333` | Vector store URL or local path |
| `--collection` | `COLLECTION` | `knomi` | Vector store collection name |

---

## Development

### Setup

```bash
# Clone and install all deps (including dev extras)
git clone https://github.com/franjofranjic27/knomi.git
cd knomi
uv sync --all-extras

# Install pre-commit hooks (run once after cloning)
uv run pre-commit install                        # commit-time hooks
uv run pre-commit install --hook-type pre-push   # push-time hooks
```

### Running tests

```bash
pytest                        # all tests
pytest tests/unit/            # unit tests only (no Docker required)
pytest tests/integration/     # requires Qdrant running
pytest -k "test_chunker"      # single test by name
```

### Infrastructure

```bash
# Start Qdrant only
docker compose up qdrant -d

# Full stack (Qdrant + ingest worker)
docker compose up
```

### Lint, format, and type check

```bash
ruff check .
ruff format .
mypy knomi
```

The pre-commit hooks run these automatically on `git commit` and `git push`.

---

## Contributing

- Commit messages follow **Conventional Commits** — see [`docs/COMMITING_CONVENTION.md`](docs/COMMITING_CONVENTION.md).
- Pull requests use the template at [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md).
- CI must be green and at least one review approval is required before merging.

For the full developer workflow (branching, releases, dependency updates) see [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md).
