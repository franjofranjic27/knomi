# ADR-004: Configurable profiles and pluggable backends

**Status:** Accepted
**Date:** 2026-07

## Context

knomi started with a single hard-wired path: Qdrant store, one embedder, one token
chunker, flat configuration. Real usage pulls in several directions at once:

- Different environments want different stores (local ChromaDB for zero-infra dev, Qdrant
  Cloud for a team, Postgres/pgvector where a Postgres already exists).
- Different budgets want different embedders (local `sentence-transformers` for free
  offline runs, OpenAI/Cohere for quality, Ollama for self-hosted).
- Different corpora want different chunking (token windows, structure/Markdown-aware,
  sentence-aware).
- Users switch between these combinations repeatedly and do not want to retype a dozen
  flags — but secrets must never end up in a checked-in config file.

Separately, we were asked whether knomi should be rewritten in TypeScript so it could ship
natively via npm, and be `brew`-installable, to reach more users.

## Options considered

### Language / distribution

| Option | Reuse of tested code | PDF quality | npm / brew reach | Notes |
|--------|---------------------|-------------|------------------|-------|
| **Keep Python, add wrappers** | full | PyMuPDF (excellent) | via thin wrappers | One CLI, one test suite |
| Rewrite CLI in TypeScript | none | weaker JS PDF libs | native | Re-implement + re-test everything |
| Dual codebases (Py + TS) | partial | mixed | native | Two implementations to keep in sync |

### Configuration shape

| Option | Multi-combo ergonomics | Secret safety | Complexity |
|--------|-----------------------|---------------|------------|
| **Nested config + JSON profiles + env secrets** | named profiles, one flag to switch | secrets only in env | moderate |
| Flat flags / env only | must retype every flag | ok | low |
| Single TOML with secrets inline | ok | secrets risk being committed | low |

## Decision

1. **Keep the pipeline in Python; distribute via thin wrappers.** PyMuPDF gives markedly
   better PDF extraction than JS alternatives, and the ingest pipeline is already
   implemented and tested. npm (`packaging/npm/`) and Homebrew (`Formula/knomi.rb`) install
   and delegate to the PyPI package rather than re-implementing the CLI. Both require
   Python ≥ 3.12.

2. **Nested configuration with named JSON profiles.** `Config` splits into `store`,
   `embedding`, and `chunking` groups. A `knomi.json` file (searched in the CWD, then
   `~/.config/knomi/`) holds named profiles, each bundling those three groups. Selection is
   `--profile` > `KNOMI_PROFILE` > the file's `default_profile`. Resolution precedence is
   **CLI flags > env (`KNOMI_`, nested via `__`) > selected profile > defaults**, wired as a
   custom `PydanticBaseSettingsSource` in `settings_customise_sources`.

3. **Secrets stay in the environment.** `OPENAI_API_KEY`, `COHERE_API_KEY`,
   `QDRANT_API_KEY`, and `KNOMI_PG_DSN` are read by `_resolve_secrets` based on the selected
   backend and are never stored in `knomi.json`.

4. **A factory per swappable part.** `build_store`, `build_embedder`, and `build_chunker`
   each read one config group and return an implementation behind an abstract base class.
   Optional dependencies are imported lazily and shipped as extras
   (`knomi[chroma]`, `[pgvector]`, `[cohere]`, `[ollama]`, `[all]`).

## Consequences

- One CLI, one test suite, one release. npm and brew add reach without a second codebase,
  at the cost of a hard Python ≥ 3.12 prerequisite on those install paths (handled with
  clear guidance in the npm postinstall/shim and a virtualenv-based formula).
- Switching environments is a single `--profile` (or `KNOMI_PROFILE`); no secrets live in
  the profile file.
- Adding a backend or strategy is a localized change: a new class plus one factory branch
  and one `Literal` entry — the pipeline and serve layers are untouched (see the extension
  points in `docs/ARCHITECTURE.md`).
- Configuration is more layered than before; the precedence order must be understood to
  reason about an effective value. It is documented in `README.md`, `knomi/config.py`, and
  the architecture doc.
- Optional backends must be installed explicitly (`knomi[...]`) or their factory branch
  raises an `ImportError` at first use — an intentional trade-off to keep the base install
  lean.
