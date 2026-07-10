# AGENTS.md

Style and workflow rules for coding agents working in this repository.
Project overview, architecture and design decisions: see [CLAUDE.md](CLAUDE.md)
and [docs/adr/](docs/adr).

## Python

- Python ≥ 3.12, dependencies managed with `uv` (lockfile `uv.lock`);
  never edit `uv.lock` by hand.
- Lint/format with ruff, types with mypy (`make lint`); both run in
  pre-commit hooks and CI — keep them green.
- Type hints on all public functions; the CLI is Typer-based
  (`typer.Argument`/`Option` defaults are intentional — ruff B008 is ignored).
- New vector stores implement the shared abstract interface (see
  ADR-001); embeddings stay pluggable.
- Tests with pytest: `tests/unit` must not require Docker;
  `tests/integration` may assume a running Qdrant (`make dev`).

## Architecture decisions

Significant decisions are recorded as ADRs in `docs/adr/ADR-NNN-<slug>.md` —
add a new ADR instead of silently changing an existing decision.

## Pull Requests

- Repo-wide standards and templates: https://github.com/franjofranjic27/.github (`REPO_STANDARDS.md`).
- Use the matching PR template from that repo (`gh pr create --body-file`):
  `dependency-update.md` for dependency updates, `sonar-fix.md` for Sonar fixes,
  `PULL_REQUEST_TEMPLATE.md` otherwise.
