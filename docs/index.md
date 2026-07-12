# knomi

knomi is a token-efficient document ingestion CLI and RAG connector for local
AI agents: it indexes your documents into a vector database (Qdrant or
ChromaDB) and serves them as a retrieval source for Claude, OpenWebUI, Ollama
and other agents.

- **Install:** `pip install knomi` ([PyPI](https://pypi.org/project/knomi/))
- **Source:** [github.com/franjofranjic27/knomi](https://github.com/franjofranjic27/knomi)
- **Usage:** see the [README](https://github.com/franjofranjic27/knomi#quick-start)

## Where to start

| Page | Content |
|---|---|
| [Architecture](ARCHITECTURE.md) | Ingestion pipeline, interfaces, serve mode |
| [ADR-001 Vector Store](adr/ADR-001-vector-store.md) | Why Qdrant default + pluggable stores |
| [ADR-002 Token Chunking](adr/ADR-002-token-chunking.md) | Why chunk sizes are measured in tokens |
| [ADR-003 SHA-256 Dedup](adr/ADR-003-sha256-dedup.md) | How unchanged documents are skipped |
| [Commit Convention](COMMIT_CONVENTION.md) | Commit message format |
| [Testing](TESTING.md) | Test types and how to run them |
| [CI/CD Workflows](WORKFLOWS.md) | GitHub Actions pipelines, releases |
| [Troubleshooting](troubleshooting.md) | Common problems and fixes |

For contributor setup see
[CONTRIBUTING.md](https://github.com/franjofranjic27/knomi/blob/main/CONTRIBUTING.md);
repo-wide standards live in
[franjofranjic27/.github](https://github.com/franjofranjic27/.github).
