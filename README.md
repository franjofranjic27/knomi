# knomi

[![CI](https://img.shields.io/github/actions/workflow/status/franjofranjic27/knomi/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/franjofranjic27/knomi/actions/workflows/ci.yml)
[![Quality Gate](https://img.shields.io/sonar/quality_gate/franjofranjic27_knomi?server=https%3A%2F%2Fsonarcloud.io&style=for-the-badge)](https://sonarcloud.io/summary/overall?id=franjofranjic27_knomi)
[![Coverage](https://img.shields.io/sonar/coverage/franjofranjic27_knomi?server=https%3A%2F%2Fsonarcloud.io&style=for-the-badge)](https://sonarcloud.io/summary/overall?id=franjofranjic27_knomi)
[![PyPI version](https://img.shields.io/pypi/v/knomi?style=for-the-badge)](https://pypi.org/project/knomi/)
[![Python](https://img.shields.io/pypi/pyversions/knomi?style=for-the-badge)](https://pypi.org/project/knomi/)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

knomi is a token-efficient document ingestion CLI and RAG connector for local
AI agents: it indexes your documents (PDF, Markdown, DOCX, HTML, plain text)
into a vector database and serves them as a retrieval source for Claude,
OpenWebUI, Ollama and other agents.

## Project Status

Published on [PyPI](https://pypi.org/project/knomi/) (`pip install knomi`);
releases are tag-driven — see [docs/WORKFLOWS.md](docs/WORKFLOWS.md).

## Features

- **Recursive document ingestion** — scans a folder for PDFs, Markdown, plain text, DOCX, and HTML files.
- **Token-efficient chunking** — splits text using `tiktoken` so `chunk_size` is always in tokens, not characters.
- **Deduplication** — stores a SHA-256 hash per source file and skips re-embedding unchanged documents.
- **Pluggable embeddings** — local models via `sentence-transformers` or OpenAI-compatible APIs.
- **Pluggable vector stores** — Qdrant (default) or ChromaDB; both share the same abstract interface.
- **RAG serve mode** — exposes the indexed vector store as an HTTP API for Claude, OpenWebUI, Ollama, and other agents.
- **Single-command infrastructure** — `docker compose up` brings up Qdrant and an optional ingest worker.

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

## Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | ≥ 3.12 | Language (uv-managed) |
| Typer | — | CLI framework |
| tiktoken | — | Token-based chunking |
| sentence-transformers | — | Local embeddings |
| Qdrant / ChromaDB | — | Vector stores |
| pytest / ruff / mypy | — | Tests, lint, types |

## Documentation

| Document | Description |
|---|---|
| [Docs site](https://franjofranjic27.github.io/knomi/) | Rendered documentation (GitHub Pages) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, pipeline, interfaces |
| [docs/adr/](docs/adr) | Architecture decision records |
| [docs/COMMIT_CONVENTION.md](docs/COMMIT_CONVENTION.md) | Commit message format and rules |
| [docs/TESTING.md](docs/TESTING.md) | How to run and write tests |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md) | GitHub Actions CI/CD workflows |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common problems and fixes |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributor setup and workflow |

Repo-wide conventions (README/badge standard, PR and issue templates) live in
[franjofranjic27/.github](https://github.com/franjofranjic27/.github).

## License

[MIT](LICENSE)
