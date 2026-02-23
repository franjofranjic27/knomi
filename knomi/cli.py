"""CLI entry point.

Subcommands
-----------
ingest  Scan a directory, embed documents, and store vectors.
serve   Expose the vector store as a RAG HTTP API.
status  Print collection info from the connected vector store.
"""

from pathlib import Path

import typer
from rich.console import Console

from knomi.config import Config

app = typer.Typer(name="knomi", help="Document ingestion and RAG connector.")
console = Console()


@app.command()
def ingest(
    source_dir: Path = typer.Argument(..., help="Directory to scan for documents."),
    chunk_size: int = typer.Option(512, help="Max chunk size in tokens."),
    chunk_overlap: int = typer.Option(64, help="Overlap between chunks in tokens."),
    embedding_model: str = typer.Option(
        "text-embedding-3-small", help="Embedding model name or HF ID."
    ),
    db_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL."),
    collection: str = typer.Option("knomi", help="Qdrant collection name."),
) -> None:
    """Scan SOURCE_DIR, embed documents, and upsert into the vector store."""
    config = Config(
        source_dir=source_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        db_url=db_url,
        collection=collection,
    )
    console.print(f"[bold]knomi ingest[/bold] — scanning [cyan]{config.source_dir}[/cyan]")
    # TODO: from knomi.ingest.pipeline import run_pipeline; run_pipeline(config)
    console.print("[yellow]Pipeline not yet implemented.[/yellow]")


@app.command()
def serve(
    db_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL."),
    collection: str = typer.Option("knomi", help="Qdrant collection name."),
    host: str = typer.Option("0.0.0.0", help="Server host."),
    port: int = typer.Option(8080, help="Server port."),
) -> None:
    """Start an HTTP server that exposes the vector store as a RAG endpoint."""
    config = Config(db_url=db_url, collection=collection, serve_host=host, serve_port=port)
    console.print(
        f"[bold]knomi serve[/bold] — [cyan]{config.serve_host}:{config.serve_port}[/cyan]"
    )
    # TODO: from knomi.serve.server import start_server; start_server(config)
    console.print("[yellow]Serve mode not yet implemented.[/yellow]")


@app.command()
def status(
    db_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL."),
    collection: str | None = typer.Option(None, help="Specific collection to inspect."),
) -> None:
    """Print collection statistics from the connected vector store."""
    config = Config(db_url=db_url)
    console.print(f"[bold]knomi status[/bold] — connecting to [cyan]{config.db_url}[/cyan]")
    # TODO: from knomi.store.qdrant import QdrantStore; QdrantStore(config).describe()
    console.print("[yellow]Status not yet implemented.[/yellow]")
