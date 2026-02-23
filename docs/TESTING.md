# Testing

## Test layout

```
tests/
├── unit/           # Pure logic — no I/O, no Docker
│   ├── test_scanner.py
│   ├── test_parser.py
│   ├── test_chunker.py
│   └── test_embedder.py
├── integration/    # Requires Qdrant running (compose or CI service)
│   └── test_pipeline.py
└── conftest.py     # Shared fixtures (tmp dirs, sample PDFs, mock embedder)
```

## Running tests

```bash
# All unit tests (fast, no infrastructure)
pytest tests/unit

# Integration tests (requires `docker compose up qdrant -d` first)
pytest tests/integration

# Single test file
pytest tests/unit/test_chunker.py

# Single test by name
pytest -k "test_chunk_respects_overlap"

# With coverage
pytest tests/unit --cov=knomi --cov-report=term-missing
```

## What to test at each layer

### Unit (`tests/unit/`)

| Module | What to cover |
|--------|---------------|
| `scanner.py` | File discovery, extension filtering, hash computation, skipping already-indexed hashes |
| `parser.py` | Each format returns plain text; whitespace normalisation; graceful handling of corrupt/empty files |
| `chunker.py` | `chunk_size` respected in tokens; overlap window correct; no empty chunks; edge cases (text shorter than chunk) |
| `embedder.py` | Batch size respected; output shape matches input; mock the actual model call |
| `config.py` | Env vars override defaults; CLI flags override env vars |

### Integration (`tests/integration/`)

| Scenario | What to verify |
|----------|----------------|
| Full ingest round-trip | File → vector in Qdrant; re-run skips unchanged files |
| Search after ingest | Top-k results returned; scores > 0 |
| Collection isolation | Two collections don't bleed into each other |

## Fixtures

`conftest.py` should provide:
- `tmp_docs_dir` — a `tmp_path` fixture pre-populated with sample `.pdf` and `.txt` files.
- `mock_embedder` — returns a fixed-dimension zero vector; avoids real model calls in unit tests.
- `qdrant_url` — reads `KNOMI_DB_URL` env var (set by CI) or falls back to `http://localhost:6333`.

## Principles

- Unit tests must run with no network access and no Docker.
- Integration tests are allowed to be slow; they run after unit tests pass.
- Do not mock the chunker or parser in integration tests — test the real path.
