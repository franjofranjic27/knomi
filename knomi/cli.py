"""CLI entry point.

Subcommands
-----------
ingest  Scan a directory, embed documents, and store vectors.
serve   Expose the vector store as a RAG HTTP API.
status  Print collection info from the connected vector store.
"""

import logging
from pathlib import Path

import typer
from rich.console import Console

from knomi.config import Config

app = typer.Typer(name="knomi", help="Document ingestion and RAG connector.")
console = Console()


@app.callback()
def _configure(
    log_level: str = typer.Option(
        "WARNING", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)."
    ),
) -> None:
    """knomi — document ingestion and RAG connector."""
    level = getattr(logging, log_level.upper(), None)
    if level is None:
        raise typer.BadParameter(f"Invalid log level: {log_level!r}")
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def ingest(
    source_dir: Path = typer.Argument(..., help="Directory to scan for documents."),
    chunk_size: int = typer.Option(512, help="Max chunk size in tokens."),
    chunk_overlap: int = typer.Option(64, help="Overlap between chunks in tokens."),
    embedding_model: str = typer.Option(
        "text-embedding-3-small", help="Embedding model name or HF ID."
    ),
    embedding_dim: int = typer.Option(1536, help="Embedding vector dimension."),
    embedding_batch_size: int = typer.Option(
        64, "--embedding-batch-size", help="Chunks per embedding API call."
    ),
    embedding_workers: int = typer.Option(
        1, "--embedding-workers", help="Parallel embedding threads (>1 for OpenAI)."
    ),
    db_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL."),
    collection: str = typer.Option("knomi", help="Qdrant collection name."),
) -> None:
    """Scan SOURCE_DIR, embed documents, and upsert into the vector store."""
    from knomi.ingest.pipeline import run_pipeline

    config = Config(
        source_dir=source_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_batch_size=embedding_batch_size,
        embedding_workers=embedding_workers,
        db_url=db_url,
        collection=collection,
    )
    console.print(f"[bold]knomi ingest[/bold] — scanning [cyan]{config.source_dir}[/cyan]")
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
    db_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL."),
    collection: str = typer.Option("knomi", help="Qdrant collection name."),
    host: str = typer.Option("0.0.0.0", help="Server host."),
    port: int = typer.Option(8080, help="Server port."),
    top_k: int = typer.Option(5, "--top-k", help="Default number of results returned per query."),
) -> None:
    """Start an HTTP server that exposes the vector store as a RAG endpoint."""
    from knomi.serve.server import start_server

    config = Config(
        db_url=db_url, collection=collection, serve_host=host, serve_port=port, top_k=top_k
    )
    console.print(
        f"[bold]knomi serve[/bold] — [cyan]{config.serve_host}:{config.serve_port}[/cyan]"
    )
    console.print(f"[dim]collection: {config.collection} · db: {config.db_url}[/dim]")
    start_server(config)


@app.command()
def status(
    db_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL."),
    collection: str | None = typer.Option(None, help="Specific collection to inspect."),
) -> None:
    """Print collection statistics from the connected vector store."""
    from rich.table import Table

    from knomi.store.qdrant import QdrantStore

    config = Config(db_url=db_url)
    if collection:
        config = Config(db_url=db_url, collection=collection)
    console.print(f"[bold]knomi status[/bold] — connecting to [cyan]{config.db_url}[/cyan]")
    info = QdrantStore(config).describe()
    table = Table(title=f"Collection: {info['name']}", show_header=False)
    table.add_row("Points", str(info["points_count"]))
    table.add_row("Indexed vectors", str(info["indexed_vectors_count"]))
    console.print(table)


@app.command()
def delete(
    doc_id: str = typer.Argument(..., help="SHA-256 hash of the document to remove."),
    db_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL."),
    collection: str = typer.Option("knomi", help="Qdrant collection name."),
) -> None:
    """Remove all vectors for a document from the collection."""
    from knomi.store.qdrant import QdrantStore

    config = Config(db_url=db_url, collection=collection)
    QdrantStore(config).delete(doc_id)
    console.print(f"[green]Deleted[/green] doc {doc_id} from {config.collection}")
