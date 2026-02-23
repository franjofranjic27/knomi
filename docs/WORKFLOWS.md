# Workflows

## First-time setup

After cloning the repository, install the pre-commit hooks so that lint, format, type checks, and unit tests run automatically before every commit and push:

```bash
uv sync --all-extras                             # install all deps including pre-commit
uv run pre-commit install                        # install commit-time hooks
uv run pre-commit install --hook-type pre-push   # install push-time hooks
```

To verify all hooks pass against the current codebase:

```bash
uv run pre-commit run --all-files
```

---

## Local development

```bash
# 1. Clone and install
git clone https://github.com/franjofranjic27/knomi.git
cd knomi
uv sync --all-extras          # installs all deps incl. dev

# 2. Start docker containers
docker compose up -d

# 3. Run the CLI against a local docs folder
python -m knomi ingest ./path/to/docs

# 4. Run tests
pytest tests/unit             # fast, no Docker
pytest tests/integration      # needs Qdrant running

# 5. Lint + format
ruff check .
ruff format .
mypy knomi
```

## CI pipeline (`.github/workflows/ci.yml`)

Triggered on every push to `main` and on every PR targeting `main`.

```
push / PR
  │
  ├─► lint    (ruff check, ruff format --check, mypy)
  │
  ├─► test    (pytest tests/unit)
  │
  └─► integration  (runs after lint+test; spins up Qdrant as a service)
```

All three jobs must pass before a PR can be merged.

## Release workflow (`.github/workflows/release.yml`)

Triggered when a tag matching `v*.*.*` is pushed.

```bash
# Cut a release
git tag v0.1.0
git push origin v0.1.0
```

The workflow builds the package with `uv build` and publishes to PyPI using OIDC trusted publishing (no API token required).

## PR process

1. Branch off `main` using the naming convention in `COMMITING_CONVENTION.md`.
2. Open a PR — the template in `.github/PULL_REQUEST_TEMPLATE.md` guides the description.
3. CI must be green.
4. At least one review approval required (enforced via `CODEOWNERS`).
5. Squash-merge or merge-commit — no rebase merges (keeps linear history easier to read).

## Dependency updates

Dependencies are pinned in `uv.lock`. To update:

```bash
uv lock --upgrade
# run tests to verify
pytest
git commit -m "chore(deps): update lockfile"
```
