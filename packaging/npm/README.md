# knomi (npm wrapper)

This npm package is a **thin wrapper** around the [`knomi`](https://pypi.org/project/knomi/)
Python package. It does **not** contain the CLI itself — installing it simply
makes the `knomi` command available via npm by delegating to the real,
PyPI-installed CLI.

## What it does

- **`postinstall`** — checks for Python >= 3.12 and best-effort installs the
  `knomi` PyPI package (`uv tool install` → `pipx install` → `pip install --user`).
  If Python is missing, it prints setup instructions instead of failing the
  `npm install`.
- **`bin/knomi.js`** — a shim that forwards all arguments and stdio to the real
  `knomi` command. If the CLI is not found, it prints how to install it.

## Requirements

- Node.js >= 18 (for the wrapper)
- Python >= 3.12 (for the actual CLI)

## Install

```bash
npm install -g knomi
knomi --help
```

## Why a wrapper?

knomi's embedding pipeline is pure Python and ships on PyPI. The npm and
Homebrew packages exist only to make installation convenient for users who
already live in those ecosystems — there is no separate JavaScript
implementation to maintain.

For the canonical install, use pip / uv / pipx directly:

```bash
uv tool install knomi
# or
pipx install knomi
```
