"""Postgres / pgvector store implementation.

Persists chunks and their embeddings in a Postgres table using the ``pgvector``
extension. Each collection maps to one table; cosine distance (``<=>``) is used
for similarity to stay consistent with the other backends.

Connection string comes from ``config.store.dsn`` (or the ``KNOMI_PG_DSN`` env
var, resolved in :func:`knomi.config._resolve_secrets`).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from knomi.config import Config
from knomi.ingest.chunker import Chunk
from knomi.store.base import SearchResult, VectorStore

log = logging.getLogger(__name__)


def _point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic id so re-indexing the same content is idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id + str(chunk_index)))


class PgVectorStore(VectorStore):
    """Postgres + pgvector backed vector store."""

    def __init__(self, config: Config) -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg import sql

        store = config.store
        if not store.dsn:
            raise ValueError("pgvector backend requires a DSN (config.store.dsn or KNOMI_PG_DSN)")
        self.dim = config.embedding.dim
        self.table = store.collection
        self._sql = sql
        self.conn = psycopg.connect(store.dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._ensure_table()

    def _tbl(self) -> Any:
        return self._sql.Identifier(self.table)

    def _ensure_table(self) -> None:
        self.conn.execute(
            self._sql.SQL(
                "CREATE TABLE IF NOT EXISTS {tbl} ("
                "  id text PRIMARY KEY,"
                "  doc_id text NOT NULL,"
                "  source text,"
                "  chunk_index integer,"
                "  total_chunks integer,"
                "  text text,"
                "  embedding vector({dim})"
                ")"
            ).format(tbl=self._tbl(), dim=self._sql.Literal(self.dim))
        )
        self.conn.execute(
            self._sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {tbl} (doc_id)").format(
                idx=self._sql.Identifier(f"{self.table}_doc_id_idx"), tbl=self._tbl()
            )
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        query = self._sql.SQL(
            "INSERT INTO {tbl} (id, doc_id, source, chunk_index, total_chunks, text, embedding)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET"
            "   doc_id = EXCLUDED.doc_id, source = EXCLUDED.source,"
            "   chunk_index = EXCLUDED.chunk_index, total_chunks = EXCLUDED.total_chunks,"
            "   text = EXCLUDED.text, embedding = EXCLUDED.embedding"
        ).format(tbl=self._tbl())
        with self.conn.cursor() as cur:
            for c, v in zip(chunks, vectors, strict=True):
                m = c.metadata
                cur.execute(
                    query,
                    (
                        _point_id(m["doc_id"], m["chunk_index"]),
                        m["doc_id"],
                        m.get("source"),
                        m.get("chunk_index"),
                        m.get("total_chunks"),
                        c.text,
                        v,
                    ),
                )

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        query = self._sql.SQL(
            "SELECT text, doc_id, source, chunk_index, total_chunks,"
            " 1 - (embedding <=> %s) AS score"
            " FROM {tbl} ORDER BY embedding <=> %s LIMIT %s"
        ).format(tbl=self._tbl())
        with self.conn.cursor() as cur:
            cur.execute(query, (vector, vector, top_k))
            rows = cur.fetchall()
        return [
            SearchResult(
                chunk=Chunk(
                    text=row[0],
                    metadata={
                        "doc_id": row[1],
                        "source": row[2],
                        "chunk_index": row[3],
                        "total_chunks": row[4],
                    },
                ),
                score=float(row[5]),
            )
            for row in rows
        ]

    def delete(self, doc_id: str) -> None:
        self.conn.execute(
            self._sql.SQL("DELETE FROM {tbl} WHERE doc_id = %s").format(tbl=self._tbl()),
            (doc_id,),
        )

    def has_document(self, doc_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                self._sql.SQL("SELECT 1 FROM {tbl} WHERE doc_id = %s LIMIT 1").format(
                    tbl=self._tbl()
                ),
                (doc_id,),
            )
            return cur.fetchone() is not None

    def get_source(self, doc_id: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute(
                self._sql.SQL("SELECT source FROM {tbl} WHERE doc_id = %s LIMIT 1").format(
                    tbl=self._tbl()
                ),
                (doc_id,),
            )
            row = cur.fetchone()
        return None if row is None else row[0]

    def update_source(self, doc_id: str, new_source: str) -> None:
        self.conn.execute(
            self._sql.SQL("UPDATE {tbl} SET source = %s WHERE doc_id = %s").format(tbl=self._tbl()),
            (new_source, doc_id),
        )

    def describe(self) -> dict[str, object]:
        with self.conn.cursor() as cur:
            cur.execute(self._sql.SQL("SELECT count(*) FROM {tbl}").format(tbl=self._tbl()))
            row = cur.fetchone()
        count = int(row[0]) if row else 0
        return {
            "name": self.table,
            "points_count": count,
            "indexed_vectors_count": count,
        }
