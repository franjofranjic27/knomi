"""Global configuration — CLI flags > env vars > .env file."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """All knomi runtime settings.

    Resolution order (highest to lowest priority):
      1. CLI flags (passed explicitly to the constructor)
      2. Environment variables prefixed with ``KNOMI_``
      3. ``.env`` file in the current working directory
      4. Defaults defined here
    """

    model_config = SettingsConfigDict(
        env_prefix="KNOMI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Ingest ---
    source_dir: Path = Field(Path("."), description="Root folder to scan for documents.")
    chunk_size: int = Field(512, gt=0, description="Maximum chunk size in tokens.")
    chunk_overlap: int = Field(64, ge=0, description="Token overlap between consecutive chunks.")

    # --- Embedding ---
    embedding_model: str = Field(
        "text-embedding-3-small",
        description="Embedding model name (OpenAI API name or HuggingFace model ID).",
    )
    embedding_dim: int = Field(
        1536, gt=0, description="Output vector dimension of the chosen model."
    )
    embedding_batch_size: int = Field(
        64, gt=0, description="Number of chunks embedded per API/model call."
    )

    # --- Vector store ---
    db_url: str = Field(
        "http://localhost:6333", description="Qdrant server URL or local file path."
    )
    collection: str = Field("knomi", description="Qdrant collection name.")

    # --- Serve ---
    serve_host: str = Field("0.0.0.0", description="Host for the RAG HTTP server.")
    serve_port: int = Field(8080, gt=0, lt=65536, description="Port for the RAG HTTP server.")
    top_k: int = Field(5, gt=0, description="Number of chunks returned per query.")
