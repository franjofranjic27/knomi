# Dockerfile for the knomi-worker service referenced by compose.yml (build: .).
# Thin image: installs only the core dependencies, no optional backends
# (chroma, pgvector, cohere, ollama). Entry point is `python -m knomi`.

FROM python:3.13-slim

# System deps: curl is only needed to fetch the uv installer.
# Cleaned up in the same layer to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast dependency resolver / installer) into /usr/local/bin.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Do not buffer stdout/stderr so logs show up immediately in `docker logs`.
ENV PYTHONUNBUFFERED=1 \
    # Install into the system environment instead of a managed venv.
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Copy only the files needed to resolve dependencies first, so this layer is
# cached and only re-run when the lockfile or project metadata changes.
COPY pyproject.toml uv.lock README.md ./

# Copy the package source (needed because hatchling builds the project itself).
COPY knomi ./knomi

# Install core dependencies only (no --all-extras), frozen to uv.lock.
RUN uv sync --frozen --no-dev

# uv sync creates a project venv at /app/.venv; put it on PATH.
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "knomi"]
CMD ["--help"]
