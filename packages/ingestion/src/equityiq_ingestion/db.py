"""Asyncpg helpers for ingestion writes.

We keep raw SQL — small surface, easier to audit, no ORM boundary issues with
pgvector. A separate retrieval package owns read paths.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg

from equityiq_ingestion.chunker import Chunk
from equityiq_ingestion.config import IngestionSettings
from equityiq_ingestion.edgar.types import FilingMeta
from equityiq_ingestion.parsers.sec_sections import Section


@dataclass(slots=True)
class FilingRow:
    id: int
    accession: str


def _format_vector(vec: Sequence[float]) -> str:
    """pgvector text input format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


class Database:
    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self._settings = settings or IngestionSettings()
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._settings.postgres_dsn,
                min_size=1,
                max_size=8,
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[asyncpg.Connection]:
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            yield conn

    async def upsert_filing(self, meta: FilingMeta) -> FilingRow:
        async with self._conn() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO filings (
                    accession, cik, ticker, form_type, filed_at,
                    period_of_report, source_url
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (accession) DO UPDATE
                  SET ticker = EXCLUDED.ticker,
                      filed_at = EXCLUDED.filed_at,
                      period_of_report = EXCLUDED.period_of_report,
                      source_url = EXCLUDED.source_url
                RETURNING id, accession
                """,
                meta.accession,
                meta.cik,
                meta.ticker,
                meta.form_type.value,
                meta.filed_at,
                meta.period_of_report,
                meta.source_url,
            )
            assert row is not None
            return FilingRow(id=row["id"], accession=row["accession"])

    async def replace_sections(
        self,
        filing_id: int,
        sections: Iterable[Section],
    ) -> dict[str, int]:
        """Delete existing sections for filing, insert new, return code→id map."""
        async with self._conn() as conn, conn.transaction():
            await conn.execute("DELETE FROM filing_sections WHERE filing_id = $1", filing_id)
            mapping: dict[str, int] = {}
            for s in sections:
                section_id = await conn.fetchval(
                    """
                    INSERT INTO filing_sections (filing_id, item_code, title, text, char_start, char_end)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                    filing_id,
                    s.item_code,
                    s.title,
                    s.text,
                    s.char_start,
                    s.char_end,
                )
                mapping[s.item_code] = int(section_id)
            await conn.execute(
                "DELETE FROM chunks WHERE filing_id = $1",
                filing_id,
            )
            return mapping

    async def insert_chunks(
        self,
        *,
        filing_id: int,
        section_id: int,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return
        rows = [
            (
                section_id,
                filing_id,
                c.ord,
                c.text,
                c.tokens,
                _format_vector(emb),
            )
            for c, emb in zip(chunks, embeddings, strict=True)
        ]
        async with self._conn() as conn:
            await conn.executemany(
                """
                INSERT INTO chunks (section_id, filing_id, ord, text, tokens, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                """,
                rows,
            )
