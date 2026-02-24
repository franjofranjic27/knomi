# Troubleshooting

## Connection errors

### `Connection refused` or `Failed to connect to Qdrant`

Qdrant is not running or the URL is wrong.

```bash
# Check if Qdrant is up
curl http://localhost:6333/healthz

# Start it
docker compose up qdrant -d

# Verify the URL in your config or env
echo $KNOMI_DB_URL   # should be http://localhost:6333
```

---

### `Collection does not exist`

knomi creates the collection automatically on first use. If you see this error, the `--db-url` or `KNOMI_DB_URL` may be pointing at the wrong Qdrant instance or collection name.

---

## Embedding errors

### `openai.AuthenticationError: No API key provided`

Set the key before running:

```bash
export OPENAI_API_KEY=sk-...
knomi ingest ./docs
```

Or add it to your `.env` file:

```
OPENAI_API_KEY=sk-...
```

---

### `openai.RateLimitError`

knomi retries automatically (up to 5 attempts, exponential backoff). If you hit sustained rate limits:

- Reduce `--embedding-batch-size` (default 64): `knomi ingest ./docs --embedding-batch-size 16`
- Set `KNOMI_EMBEDDING_WORKERS=1` to disable concurrent batching.

---

### `sentence_transformers` model download is slow or fails

Local models are downloaded from Hugging Face on first use. Set a custom cache dir:

```bash
export HF_HOME=/path/to/model/cache
```

Or pre-download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

---

## Ingest issues

### Files are silently skipped

knomi skips files that are already indexed (same SHA-256 hash). Run with `--log-level INFO` to see which files are being skipped:

```bash
knomi --log-level INFO ingest ./docs
```

---

### `Empty text extracted from <file>`

The parser returned no text. Common causes:

- **PDF is image-only** (scanned document): PyMuPDF cannot extract text without OCR. Consider running OCR preprocessing first (e.g. `ocrmypdf`).
- **DOCX is corrupted** or password-protected.
- **HTML has no visible text** (JavaScript-rendered content): knomi parses static HTML only.

---

### Move detection not triggering

If a file is re-indexed after being moved (instead of just updating metadata), the old and new files may have different content (e.g. timestamp in headers). Check with:

```bash
sha256sum old/path/file.pdf new/path/file.pdf
```

If the hashes differ, it is a new document, not a move.

---

## Serve mode issues

### `POST /query` returns empty results

The collection may be empty. Check with:

```bash
knomi status
```

If `points_count` is 0, run `knomi ingest` first.

---

### `POST /query` is slow

- The embedding model is loaded on every server start. Subsequent requests reuse the loaded model.
- For local models, the first request triggers model loading (~1–5 s). Subsequent requests are fast.
- For OpenAI embeddings, latency depends on API response time (~100–300 ms).

---

## Debugging

Enable debug logging to see per-file processing details:

```bash
knomi --log-level DEBUG ingest ./docs 2>&1 | head -100
```

Run unit tests to verify the pipeline is intact:

```bash
make test
```

Run integration tests against a live Qdrant instance:

```bash
make dev          # starts Qdrant
make test-all
```
