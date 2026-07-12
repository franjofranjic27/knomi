"""Global configuration — CLI flags > env vars > JSON profile > defaults.

Configuration is split into three nested groups (``store``, ``embedding``,
``chunking``) plus a few top-level fields. Values are resolved from four
layers, highest priority first:

  1. CLI flags (passed explicitly to :func:`resolve_config`)
  2. Environment variables prefixed with ``KNOMI_`` (nested via ``__``,
     e.g. ``KNOMI_STORE__COLLECTION``)
  3. The selected profile in ``knomi.json`` (see :func:`load_profiles`)
  4. Defaults defined here

Secrets are never stored in ``knomi.json`` — the profile only references them
by convention; the actual values are read from the environment in
:func:`_resolve_secrets` (``OPENAI_API_KEY``, ``COHERE_API_KEY``,
``QDRANT_API_KEY``, ``KNOMI_PG_DSN``).
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Name of the profile file searched in the CWD and the user config dir.
PROFILES_FILENAME = "knomi.json"

# Active profile name for the current resolution. Set by resolve_config() so the
# profile settings source (below) knows which profile block to load. A ContextVar
# keeps it isolated per async/thread context.
_ACTIVE_PROFILE: ContextVar[str | None] = ContextVar("knomi_active_profile", default=None)


# --------------------------------------------------------------------------- #
# Nested settings groups
# --------------------------------------------------------------------------- #
class StoreSettings(BaseModel):
    """Vector store connection settings."""

    backend: Literal["qdrant", "chroma", "pgvector"] = Field(
        "qdrant", description="Vector store backend to use."
    )
    url: str = Field(
        "http://localhost:6333",
        description="Qdrant server URL (http/https/grpc) or local path.",
    )
    path: str | None = Field(
        None, description="On-disk path for local backends (ChromaDB / local Qdrant)."
    )
    api_key: str | None = Field(
        None, description="API key for managed backends (e.g. Qdrant Cloud)."
    )
    dsn: str | None = Field(None, description="Postgres DSN for the pgvector backend.")
    collection: str = Field("knomi", description="Collection / table name.")


class EmbeddingSettings(BaseModel):
    """Embedding backend settings."""

    backend: Literal["openai", "local", "cohere", "ollama"] = Field(
        "openai", description="Embedding backend to use."
    )
    model: str = Field(
        "text-embedding-3-small", description="Model name (API name or HF/Ollama ID)."
    )
    dim: int = Field(1536, gt=0, description="Output vector dimension of the model.")
    batch_size: int = Field(64, gt=0, description="Texts embedded per backend call.")
    workers: int = Field(1, gt=0, description="Parallel workers for batch embedding.")
    api_key: str | None = Field(None, description="API key for API-backed embedders.")
    host: str | None = Field(None, description="Base URL for self-hosted backends (e.g. Ollama).")


class ChunkingSettings(BaseModel):
    """Chunking strategy settings."""

    strategy: Literal["token", "structure", "sentence"] = Field(
        "token", description="Chunking strategy."
    )
    chunk_size: int = Field(512, gt=0, description="Maximum chunk size in tokens.")
    chunk_overlap: int = Field(64, ge=0, description="Token overlap between chunks.")
    encoding_name: str = Field(
        "cl100k_base", description="tiktoken encoding used for token counting."
    )


class RerankingSettings(BaseModel):
    """Cross-encoder reranking settings (opt-in second retrieval stage)."""

    enabled: bool = Field(False, description="Re-score candidates with a cross-encoder.")
    backend: Literal["local", "cohere"] = Field("local", description="Reranker backend.")
    model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model (HF CrossEncoder id or Cohere rerank model).",
    )
    top_n: int = Field(20, gt=0, description="Candidates fetched from the store before reranking.")
    api_key: str | None = Field(None, description="API key for API-backed rerankers.")


# --------------------------------------------------------------------------- #
# Profile settings source
# --------------------------------------------------------------------------- #
def _profiles_search_paths() -> list[Path]:
    """Return candidate locations for ``knomi.json`` (highest priority first)."""
    paths = [Path.cwd() / PROFILES_FILENAME]
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    paths.append(base / "knomi" / PROFILES_FILENAME)
    return paths


def load_profiles() -> dict[str, Any]:
    """Load and return the parsed ``knomi.json`` document, or ``{}`` if absent.

    The document has the shape ``{"default_profile": str, "profiles": {name: {...}}}``.
    """
    for path in _profiles_search_paths():
        if path.is_file():
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"{path}: expected a JSON object at the top level")
            return data
    return {}


def resolve_profile_name(profile: str | None) -> str | None:
    """Return the effective profile name (explicit > env > file default)."""
    return profile or os.environ.get("KNOMI_PROFILE") or load_profiles().get("default_profile")


class _ProfileSettingsSource(PydanticBaseSettingsSource):
    """pydantic-settings source that injects the selected ``knomi.json`` profile.

    Placed below the env/dotenv sources so environment variables override the
    profile, but above field defaults so the profile overrides them.
    """

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:  # noqa: D102
        # All work happens in __call__; per-field extraction is unused.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        name = resolve_profile_name(_ACTIVE_PROFILE.get())
        if not name:
            return {}
        data = load_profiles()
        profiles = data.get("profiles", {})
        if name not in profiles:
            available = ", ".join(sorted(profiles)) or "none"
            raise KeyError(f"Profile {name!r} not found in knomi.json (available: {available})")
        profile = profiles[name]
        if not isinstance(profile, dict):
            raise ValueError(f"Profile {name!r} must be a JSON object")
        return profile


# --------------------------------------------------------------------------- #
# Top-level configuration
# --------------------------------------------------------------------------- #
class Config(BaseSettings):
    """All knomi runtime settings."""

    model_config = SettingsConfigDict(
        env_prefix="KNOMI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Ingest ---
    source_dir: Path = Field(Path("."), description="Root folder to scan for documents.")

    # --- Nested groups ---
    store: StoreSettings = Field(default_factory=StoreSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    reranking: RerankingSettings = Field(default_factory=RerankingSettings)

    # --- Serve ---
    serve_host: str = Field("0.0.0.0", description="Host for the RAG HTTP server.")
    serve_port: int = Field(8080, gt=0, lt=65536, description="Port for the RAG HTTP server.")
    top_k: int = Field(5, gt=0, description="Number of chunks returned per query.")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority order (highest first): CLI/init > env > .env > JSON profile > defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _ProfileSettingsSource(settings_cls),
        )


def _resolve_secrets(config: Config) -> None:
    """Fill in unset secrets from conventional environment variables (in place)."""
    if config.embedding.api_key is None:
        if config.embedding.backend == "openai":
            config.embedding.api_key = os.environ.get("OPENAI_API_KEY")
        elif config.embedding.backend == "cohere":
            config.embedding.api_key = os.environ.get("COHERE_API_KEY")
    if config.store.api_key is None and config.store.backend == "qdrant":
        config.store.api_key = os.environ.get("QDRANT_API_KEY")
    if config.store.dsn is None and config.store.backend == "pgvector":
        config.store.dsn = os.environ.get("KNOMI_PG_DSN")
    if (
        config.reranking.api_key is None
        and config.reranking.enabled
        and config.reranking.backend == "cohere"
    ):
        config.reranking.api_key = os.environ.get("COHERE_API_KEY")


def resolve_config(
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Build a :class:`Config` honouring CLI > env > JSON profile > defaults.

    Args:
        profile:   Profile name to select from ``knomi.json``. Falls back to
                   ``KNOMI_PROFILE`` and then the file's ``default_profile``.
        overrides: Nested dict of explicit CLI overrides (highest priority).
                   Keys with ``None`` values are dropped so unset flags do not
                   clobber lower-priority layers.

    Returns:
        A fully resolved ``Config`` with secrets filled in from the environment.
    """
    clean = _strip_none(overrides or {})
    token = _ACTIVE_PROFILE.set(profile)
    try:
        config = Config(**clean)
    finally:
        _ACTIVE_PROFILE.reset(token)
    _resolve_secrets(config)
    return config


def _strip_none(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively drop keys whose value is ``None`` (and empty sub-dicts)."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested = _strip_none(value)
            if nested:
                result[key] = nested
        else:
            result[key] = value
    return result
