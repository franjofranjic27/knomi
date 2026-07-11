# knomi

> Token-efficient document ingestion and RAG connector for local AI agents.

[![CI](https://github.com/franjofranjic27/knomi/actions/workflows/ci.yml/badge.svg)](https://github.com/franjofranjic27/knomi/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/knomi)](https://pypi.org/project/knomi/)
[![Python](https://img.shields.io/pypi/pyversions/knomi)](https://pypi.org/project/knomi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- **Recursive document ingestion** — scans a folder for PDFs, Markdown, plain text, DOCX, and HTML files.
- **Token-efficient chunking** — expresses `chunk_size`/`chunk_overlap` in tokens (via `tiktoken`), not characters, across three strategies: `token`, `structure`, `sentence`.
- **Deduplication** — stores a SHA-256 hash per source file and skips re-embedding unchanged documents.
- **Pluggable embeddings** — `openai`, `local` (`sentence-transformers`), `cohere`, or `ollama`.
- **Pluggable vector stores** — `qdrant` (default), `chroma`, or `pgvector`; all share the same abstract interface.
- **Named profiles** — bundle store, embedding, and chunking settings in `knomi.json` and switch between them with `--profile`.
- **RAG serve mode** — exposes the indexed vector store as an HTTP API for Claude, OpenWebUI, Ollama, and other agents.
- **Single-command infrastructure** — `docker compose up` brings up Qdrant and an optional ingest worker.

---

## Installation

knomi is a Python package published on PyPI. The npm and Homebrew options are thin
wrappers that install and delegate to that same package — they **require Python ≥ 3.12**
on the machine.

```bash
# PyPI (recommended)
pip install knomi                # or: uv tool install knomi / pipx install knomi

# npm — wrapper; runs `pip/pipx/uv install knomi` on postinstall (needs Python ≥ 3.12)
npm install -g knomi

# Homebrew — formula builds an isolated virtualenv (needs Python ≥ 3.12)
brew tap franjofranjic27/knomi
brew install knomi
```

Optional backends ship as extras and are imported lazily — install only what you use:

```bash
pip install "knomi[chroma]"      # ChromaDB store
pip install "knomi[pgvector]"    # Postgres/pgvector store
pip install "knomi[cohere]"      # Cohere embeddings
pip install "knomi[ollama]"      # Ollama embeddings
pip install "knomi[all]"         # everything at once
```

---

## Quick start

### Option A — Docker + Qdrant (recommended)

```bash
# 1. Clone the repo (for compose.yml) and start Qdrant
git clone https://github.com/franjofranjic27/knomi.git
cd knomi
docker compose up qdrant -d

# 2. Install knomi and ingest your documents
pip install knomi
knomi ingest ./docs --backend qdrant --db-url http://localhost:6333 --collection my-kb
```

### Option B — zero infrastructure (ChromaDB)

```bash
pip install "knomi[chroma]"
knomi ingest ./docs --backend chroma --db-url ./.knomi/chroma --collection my-kb
```

---

## CLI usage

The global `--profile` / `-p` option selects a profile from `knomi.json` (see
[Profiles & configuration](#profiles--configuration)) and applies to every subcommand.

### `ingest` — index documents into the vector store

```bash
# Ingest a folder using the default (or selected) profile
knomi ingest ./docs

# Pick a profile and override the chunking strategy
knomi ingest ./docs --profile cloud --strategy structure

# Custom chunk size, overlap, and collection
knomi ingest ./docs --chunk-size 512 --chunk-overlap 64 --collection my-kb

# Choose store + embedding backends explicitly
knomi ingest ./docs --backend qdrant --db-url http://qdrant:6333 \
  --embedding-backend openai --embedding-model text-embedding-3-small --embedding-dim 1536
```

### `profiles` — list the profiles defined in `knomi.json`

```bash
knomi profiles
# Shows each profile's store, embedding, and chunking backends; marks the default.
```

### `status` — inspect a collection

```bash
knomi status
# Prints point count and indexed-vector count for the resolved collection.
```

### `delete` — remove a document from a collection

```bash
knomi delete <sha256-doc-id>
# Removes every vector whose doc_id matches the given SHA-256.
```

### `serve` — expose RAG as an HTTP API for agents

```bash
knomi serve --port 8080
# Starts an HTTP server (GET /health, POST /query) that agents can query.
```

### `eval` — measure retrieval quality against a gold set

Turn "did I pick the right embedding model / chunking strategy?" from a guess
into a measurement. A **gold set** (JSON or JSONL) lists questions and the source
documents that *should* be retrieved; `knomi eval` reports Recall@k, nDCG@k,
Hit@k and MRR for the selected profile against an already-indexed collection.

```bash
# Gold set (JSONL — one question per line, document-level relevance):
#   {"question": "What is self-attention?", "relevant_sources": ["Transformer.pdf"]}

knomi --profile local eval gold.jsonl --collection uni-sg --top-k 10
knomi --profile local eval gold.jsonl --json report.json   # machine-readable dump
```

Because each profile is a full config (embedding × chunking × store), evaluating
is an A/B test — run the same gold set under `-p local` vs `-p cloud` and compare
the numbers. See `eval.example.jsonl` for a template. `relevant_sources` are
matched leniently (basename / substring), so you only need the file name.

### Reranking (optional second stage)

Vector search is fast but scores query and document independently. A **cross-encoder
reranker** re-scores the top candidates jointly, which is often (but not always!)
more accurate. It is opt-in per profile:

```json
"reranking": {
  "enabled": true,
  "backend": "local",
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "top_n": 20
}
```

The store returns `top_n` candidates, the reranker re-scores them down to your
`top_k`. Backends: `local` (HuggingFace `CrossEncoder`, no API key) or `cohere`.
Toggle it ad hoc for A/B testing:

```bash
knomi -p local eval gold.jsonl --collection uni-sg              # baseline
knomi -p local eval gold.jsonl --collection uni-sg --rerank     # with reranking
```

> Reranking is **not** a guaranteed win — a reranker trained on a different domain,
> or a small gold set, can regress. Measure it with `knomi eval` before enabling it.

---

## Profiles & configuration

### Configuration precedence

Settings resolve from four layers, highest priority first:

1. **CLI flags** — e.g. `--backend`, `--strategy`, `--collection`.
2. **Environment variables** — prefixed with `KNOMI_`, nested groups joined by `__`
   (e.g. `KNOMI_STORE__COLLECTION`, `KNOMI_EMBEDDING__BACKEND`).
3. **The selected `knomi.json` profile** — chosen via `--profile`, `KNOMI_PROFILE`, or the
   file's `default_profile`.
4. **Built-in defaults**.

### `knomi.json`

A profile bundles the three nested config groups (`store`, `embedding`, `chunking`). knomi
searches for `knomi.json` in the current working directory first, then in
`~/.config/knomi/` (`$XDG_CONFIG_HOME/knomi/` if set). Copy
[`knomi.example.json`](knomi.example.json) to get started.

```json
{
  "default_profile": "local",
  "profiles": {
    "local": {
      "store": { "backend": "chroma", "path": "./.knomi/chroma", "collection": "knomi" },
      "embedding": { "backend": "local", "model": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384 },
      "chunking": { "strategy": "structure", "chunk_size": 512, "chunk_overlap": 64 }
    },
    "cloud": {
      "store": { "backend": "qdrant", "url": "https://your-cluster.qdrant.io:6333", "collection": "knomi" },
      "embedding": { "backend": "openai", "model": "text-embedding-3-small", "dim": 1536, "workers": 4 },
      "chunking": { "strategy": "token", "chunk_size": 512, "chunk_overlap": 64 }
    }
  }
}
```

Select a profile per invocation:

```bash
knomi --profile cloud ingest ./docs      # or: KNOMI_PROFILE=cloud knomi ingest ./docs
```

### Secrets

Secrets are **never** stored in `knomi.json`. They are read from the environment and merged
into the resolved config based on the selected backend:

| Env var | Used by |
|---------|---------|
| `OPENAI_API_KEY` | `embedding.backend = openai` |
| `COHERE_API_KEY` | `embedding.backend = cohere` |
| `QDRANT_API_KEY` | `store.backend = qdrant` (Qdrant Cloud) |
| `KNOMI_PG_DSN` | `store.backend = pgvector` |

### Backends & strategies

| Group | Config key | Values | Notes |
|-------|-----------|--------|-------|
| Store | `store.backend` | `qdrant` (default), `chroma`, `pgvector` | `chroma`/`pgvector` need the matching extra |
| Embedding | `embedding.backend` | `openai` (default), `local`, `cohere`, `ollama` | `cohere`/`ollama` need the matching extra |
| Chunking | `chunking.strategy` | `token` (default), `structure`, `sentence` | all express sizes in tokens |

### Selected fields

| Config key | Env var | Default | Description |
|-----------|---------|---------|-------------|
| `source_dir` | `KNOMI_SOURCE_DIR` | `.` | Folder to scan for documents |
| `store.backend` | `KNOMI_STORE__BACKEND` | `qdrant` | Vector store backend |
| `store.url` | `KNOMI_STORE__URL` | `http://localhost:6333` | Qdrant server URL |
| `store.path` | `KNOMI_STORE__PATH` | `None` | On-disk path for local backends |
| `store.collection` | `KNOMI_STORE__COLLECTION` | `knomi` | Collection / table name |
| `embedding.backend` | `KNOMI_EMBEDDING__BACKEND` | `openai` | Embedding backend |
| `embedding.model` | `KNOMI_EMBEDDING__MODEL` | `text-embedding-3-small` | Model name or ID |
| `embedding.dim` | `KNOMI_EMBEDDING__DIM` | `1536` | Output vector dimension |
| `chunking.strategy` | `KNOMI_CHUNKING__STRATEGY` | `token` | Chunking strategy |
| `chunking.chunk_size` | `KNOMI_CHUNKING__CHUNK_SIZE` | `512` | Max chunk size in tokens |
| `chunking.chunk_overlap` | `KNOMI_CHUNKING__CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `serve_port` | `KNOMI_SERVE_PORT` | `8080` | Port for the RAG HTTP server |
| `top_k` | `KNOMI_TOP_K` | `5` | Chunks returned per query |

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
