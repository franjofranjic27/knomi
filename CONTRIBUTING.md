# Contributing to knomi

## Setup

**Prerequisites:** Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), Docker with Compose v2

```bash
git clone https://github.com/franjofranjic27/knomi.git
cd knomi
uv sync --all-extras

# Install pre-commit hooks (run once after cloning)
uv run pre-commit install                        # commit-time hooks
uv run pre-commit install --hook-type pre-push   # push-time hooks
```

## Development

```bash
# Infrastructure
docker compose up qdrant -d   # Qdrant only
docker compose up             # full stack (Qdrant + ingest worker)

# Tests
pytest                        # all tests
pytest tests/unit/            # unit tests only (no Docker required)
pytest tests/integration/     # requires Qdrant running

# Lint, format, type check (also enforced by the pre-commit hooks)
ruff check . && ruff format . && mypy knomi
```

`make lint`, `make test`, `make test-all`, `make dev` wrap the same commands —
see the [Makefile](Makefile).

## Commits & pull requests

- Commit messages follow **Conventional Commits** — see
  [docs/COMMIT_CONVENTION.md](docs/COMMIT_CONVENTION.md).
- One logical change per commit; PRs use the templates from
  [franjofranjic27/.github](https://github.com/franjofranjic27/.github)
  (`dependency-update.md` for dependency PRs, `sonar-fix.md` for Sonar fixes).
- CI and SonarCloud must be green before merge.

## Releases

Tag-driven, published to PyPI — see [docs/WORKFLOWS.md](docs/WORKFLOWS.md).
