"""HTTP server for the knomi RAG query API.

Endpoints
---------
GET  /health   Liveness check — returns collection name and status.
POST /query    Embed a query string and return the top-k most similar chunks.
GET  /openapi.json  Auto-generated OpenAPI schema (served by FastAPI).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from knomi.config import Config
from knomi.ingest.embedder import build_embedder
from knomi.store.qdrant import QdrantStore


class QueryRequest(BaseModel):
    """Payload for a semantic search request."""

    query: str = Field(..., description="Natural-language query string.")
    top_k: int = Field(5, gt=0, description="Number of chunks to return.")


class ChunkResult(BaseModel):
    """A single retrieved chunk with its similarity score."""

    text: str = Field(..., description="Chunk plain text.")
    score: float = Field(..., description="Cosine similarity score (0–1).")
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response envelope for a /query request."""

    results: list[ChunkResult]
    collection: str = Field(..., description="Qdrant collection that was queried.")


def create_app(config: Config) -> FastAPI:
    """Build and return the FastAPI application.

    The embedder and store are initialised eagerly so the app is ready to
    serve requests as soon as this function returns.

    Args:
        config: Runtime configuration (DB URL, collection, embedding model, etc.).

    Returns:
        A configured FastAPI application instance.
    """
    embedder = build_embedder(config)
    store = QdrantStore(config)

    app = FastAPI(
        title="knomi RAG API",
        description=(
            "Query the knomi vector store for semantically relevant document chunks. "
            "Run `knomi ingest <dir>` first to populate the collection."
        ),
        version="0.1.0",
    )

    @app.get("/health", summary="Liveness check")
    def health() -> dict[str, str]:
        """Return OK and the active collection name."""
        return {"status": "ok", "collection": config.collection}

    @app.post("/query", response_model=QueryResponse, summary="Semantic search")
    def query(req: QueryRequest) -> QueryResponse:
        """Embed *req.query* and return the *req.top_k* most similar chunks."""
        vector = embedder.embed_query(req.query)
        hits = store.search(vector, top_k=req.top_k)
        return QueryResponse(
            results=[
                ChunkResult(text=h.chunk.text, score=h.score, metadata=h.chunk.metadata)
                for h in hits
            ],
            collection=config.collection,
        )

    return app


def start_server(config: Config) -> None:
    """Start the uvicorn ASGI server (blocking).

    Args:
        config: Runtime configuration — ``serve_host`` and ``serve_port`` are used.
    """
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=config.serve_host, port=config.serve_port)
