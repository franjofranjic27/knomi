"""CLI entry point.

Subcommands
-----------
ingest    Scan a directory, embed documents, and store vectors.
serve     Expose the vector store as a RAG HTTP API.
status    Print collection info from the connected vector store.
delete    Remove a document's vectors from the collection.
profiles  List the profiles defined in knomi.json.

Configuration precedence: CLI flags > env (``KNOMI_``) > selected ``knomi.json``
profile > defaults. Unset flags default to ``None`` so they never clobber the
lower-priority layers. Select a profile with the global ``--profile`` option.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from knomi.config import resolve_config

app = typer.Typer(name="knomi", help="Document ingestion and RAG connector.")
console = Console()


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None (resolve_config also strips recursively)."""
    return {k: v for k, v in data.items() if v is not None}


@app.callback()
def _configure(
    ctx: typer.Context,
    profile: str | None = typer.Option(
        None, "--profile", "-p", help="Profile name from knomi.json."
    ),
    log_level: str = typer.Option(
        "WARNING", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)."
    ),
) -> None:
    """knomi — document ingestion and RAG connector."""
    level = getattr(logging, log_level.upper(), None)
    if level is None:
        raise typer.BadParameter(f"Invalid log level: {log_level!r}")
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    ctx.obj = {"profile": profile}


def _profile(ctx: typer.Context) -> str | None:
    return (ctx.obj or {}).get("profile")


@app.command()
def ingest(
    ctx: typer.Context,
    source_dir: Path = typer.Argument(..., help="Directory to scan for documents."),
    strategy: str | None = typer.Option(None, help="Chunking strategy: token|structure|sentence."),
    chunk_size: int | None = typer.Option(None, help="Max chunk size in tokens."),
    chunk_overlap: int | None = typer.Option(None, help="Overlap between chunks in tokens."),
    embedding_backend: str | None = typer.Option(
        None, "--embedding-backend", help="Embedding backend: openai|local|cohere|ollama."
    ),
    embedding_model: str | None = typer.Option(None, help="Embedding model name or ID."),
    embedding_dim: int | None = typer.Option(None, help="Embedding vector dimension."),
    backend: str | None = typer.Option(None, help="Store backend: qdrant|chroma|pgvector."),
    db_url: str | None = typer.Option(None, "--db-url", help="Vector DB server URL or path."),
    collection: str | None = typer.Option(None, help="Collection / table name."),
) -> None:
    """Scan SOURCE_DIR, embed documents, and upsert into the vector store."""
    from knomi.ingest.pipeline import run_pipeline

    config = resolve_config(
        profile=_profile(ctx),
        overrides={
            "source_dir": source_dir,
            "chunking": _clean(
                {"strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            ),
            "embedding": _clean(
                {"backend": embedding_backend, "model": embedding_model, "dim": embedding_dim}
            ),
            "store": _clean({"backend": backend, "url": db_url, "collection": collection}),
        },
    )
    console.print(f"[bold]knomi ingest[/bold] — scanning [cyan]{config.source_dir}[/cyan]")
    console.print(
        f"[dim]store: {config.store.backend} · embed: {config.embedding.backend}"
        f" · chunk: {config.chunking.strategy}[/dim]"
    )
    result = run_pipeline(config, progress_callback=lambda msg: console.log(msg))
    parts = [
        f"{result.total_files} files",
        f"{result.total_chunks} chunks",
        f"{result.total_vectors} vectors",
        f"skipped {result.skipped_files}",
    ]
    if result.moved_files:
        parts.append(f"moved {result.moved_files}")
    parts.append(f"failed {len(result.failed_files)}")
    console.print(f"[green]Done![/green] {' · '.join(parts)}")


@app.command()
def serve(
    ctx: typer.Context,
    backend: str | None = typer.Option(None, help="Store backend: qdrant|chroma|pgvector."),
    db_url: str | None = typer.Option(None, "--db-url", help="Vector DB server URL or path."),
    collection: str | None = typer.Option(None, help="Collection / table name."),
    host: str | None = typer.Option(None, help="Server host."),
    port: int | None = typer.Option(None, help="Server port."),
    top_k: int | None = typer.Option(None, "--top-k", help="Default results per query."),
) -> None:
    """Start an HTTP server that exposes the vector store as a RAG endpoint."""
    from knomi.serve.server import start_server

    config = resolve_config(
        profile=_profile(ctx),
        overrides={
            "store": _clean({"backend": backend, "url": db_url, "collection": collection}),
            "serve_host": host,
            "serve_port": port,
            "top_k": top_k,
        },
    )
    console.print(
        f"[bold]knomi serve[/bold] — [cyan]{config.serve_host}:{config.serve_port}[/cyan]"
    )
    console.print(
        f"[dim]collection: {config.store.collection} · store: {config.store.backend}[/dim]"
    )
    start_server(config)


@app.command()
def status(
    ctx: typer.Context,
    backend: str | None = typer.Option(None, help="Store backend: qdrant|chroma|pgvector."),
    db_url: str | None = typer.Option(None, "--db-url", help="Vector DB server URL or path."),
    collection: str | None = typer.Option(None, help="Specific collection to inspect."),
) -> None:
    """Print collection statistics from the connected vector store."""
    from rich.table import Table

    from knomi.store.factory import build_store, store_target

    config = resolve_config(
        profile=_profile(ctx),
        overrides={"store": _clean({"backend": backend, "url": db_url, "collection": collection})},
    )
    console.print(
        f"[bold]knomi status[/bold] — {config.store.backend} @ "
        f"[cyan]{store_target(config.store)}[/cyan]"
    )
    info = build_store(config).describe()
    table = Table(title=f"Collection: {info['name']}", show_header=False)
    table.add_row("Points", str(info["points_count"]))
    table.add_row("Indexed vectors", str(info["indexed_vectors_count"]))
    console.print(table)


@app.command()
def delete(
    ctx: typer.Context,
    doc_id: str = typer.Argument(..., help="SHA-256 hash of the document to remove."),
    backend: str | None = typer.Option(None, help="Store backend: qdrant|chroma|pgvector."),
    db_url: str | None = typer.Option(None, "--db-url", help="Vector DB server URL or path."),
    collection: str | None = typer.Option(None, help="Collection / table name."),
) -> None:
    """Remove all vectors for a document from the collection."""
    from knomi.store.factory import build_store

    config = resolve_config(
        profile=_profile(ctx),
        overrides={"store": _clean({"backend": backend, "url": db_url, "collection": collection})},
    )
    build_store(config).delete(doc_id)
    console.print(f"[green]Deleted[/green] doc {doc_id} from {config.store.collection}")


@app.command()
def profiles() -> None:
    """List the profiles defined in knomi.json."""
    from rich.table import Table

    from knomi.config import load_profiles

    data = load_profiles()
    if not data or not data.get("profiles"):
        console.print("[yellow]No knomi.json found[/yellow] (or it defines no profiles).")
        console.print("Copy [cyan]knomi.example.json[/cyan] to [cyan]knomi.json[/cyan] to start.")
        raise typer.Exit()

    default = data.get("default_profile")
    table = Table(title="knomi profiles")
    table.add_column("Profile")
    table.add_column("Store")
    table.add_column("Embedding")
    table.add_column("Chunking")
    for name, prof in data["profiles"].items():
        label = f"{name} [green](default)[/green]" if name == default else name
        store = prof.get("store", {})
        embedding = prof.get("embedding", {})
        chunking = prof.get("chunking", {})
        table.add_row(
            label,
            f"{store.get('backend', '-')}",
            f"{embedding.get('backend', '-')}:{embedding.get('model', '-')}",
            f"{chunking.get('strategy', '-')}",
        )
    console.print(table)
