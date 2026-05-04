"""Hybrid retrieval: pgvector (dense) + Postgres FTS (lexical) → RRF → rerank.

Pipeline per query:
    1. Embed query with same model as chunks (nomic-embed-text-v1.5, 768d).
    2. Issue two SQL queries in parallel:
         dense   : ORDER BY embedding <=> $vec  (cosine distance, ASC)
         lexical : WHERE plainto_tsquery matches; ORDER BY ts_rank_cd DESC
       Each returns top `pool` candidates.
    3. Reciprocal Rank Fusion → reorder candidates.
    4. Optional rerank step: cross-encoder over top `pool_for_rerank`.
    5. Return top `final_top_k`.

Filters supported: ticker, form_type, item_code (only-1A, only-MD&A, etc.),
filed_after.

Postgres index choices:
    - HNSW on chunks.embedding (created in init.sql, m=16, ef_construction=64).
    - GIN on to_tsvector('english', text).

Notes:
    - We use "chunks.embedding <=> $1::vector" → cosine distance; lower is
      better. We expose 1 - distance as `dense_score` so callers see "higher
      is better" everywhere.
    - We DON'T fuse on raw scores; RRF on ranks only (avoids pgvector vs.
      ts_rank_cd scale mismatch issues).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import asyncpg
from equityiq_llm import OllamaClient
from equityiq_observability import observed
from pydantic import BaseModel, Field

from equityiq_retrieval.config import RetrievalSettings
from equityiq_retrieval.fusion import reciprocal_rank_fusion
from equityiq_retrieval.reranker import TEIReranker
from equityiq_retrieval.types import RetrievalResult


def _format_vector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


class RetrievalQuery(BaseModel):
    query: str = Field(..., min_length=2)
    ticker: str | None = None
    form_types: list[str] | None = None
    item_codes: list[str] | None = None
    filed_after: str | None = None  # ISO date
    top_k: int | None = None  # override settings.final_top_k
    pool: int | None = None  # override settings.candidate_pool_size
    use_reranker: bool = True


@dataclass(slots=True)
class _Candidate:
    chunk_id: int
    filing_id: int
    section_id: int
    item_code: str
    text: str
    ticker: str | None
    accession: str
    form_type: str
    filed_at: object
    source_url: str
    chunk_ord: int
    dense_score: float | None = None
    lexical_score: float | None = None


_BASE_SELECT = """
    c.id              AS chunk_id,
    c.filing_id       AS filing_id,
    c.section_id      AS section_id,
    c.ord             AS chunk_ord,
    c.text            AS text,
    s.item_code       AS item_code,
    f.ticker          AS ticker,
    f.accession       AS accession,
    f.form_type       AS form_type,
    f.filed_at        AS filed_at,
    f.source_url      AS source_url
"""


class HybridRetriever:
    def __init__(
        self,
        *,
        llm: OllamaClient,
        reranker: TEIReranker | None = None,
        settings: RetrievalSettings | None = None,
        pool: asyncpg.Pool | None = None,
    ) -> None:
        self._llm = llm
        self._reranker = reranker
        self._settings = settings or RetrievalSettings()
        self._pool = pool
        self._owns_pool = pool is None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._settings.postgres_dsn, min_size=1, max_size=8
            )

    async def aclose(self) -> None:
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[asyncpg.Connection]:
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as c:
            yield c

    @observed("retrieval.search")
    async def search(self, q: RetrievalQuery) -> list[RetrievalResult]:
        pool_size = q.pool or self._settings.candidate_pool_size
        top_k = q.top_k or self._settings.final_top_k

        embedding = (await self._llm.embed([q.query]))[0]

        dense_task = asyncio.create_task(self._dense(q, embedding, pool_size))
        lexical_task = asyncio.create_task(self._lexical(q, pool_size))
        dense_cands, lexical_cands = await asyncio.gather(dense_task, lexical_task)

        merged = self._merge(dense_cands, lexical_cands)
        fused = reciprocal_rank_fusion(
            [
                [c.chunk_id for c in dense_cands],
                [c.chunk_id for c in lexical_cands],
            ],
            k=self._settings.rrf_k,
        )

        ordered = sorted(merged.values(), key=lambda c: fused.get(c.chunk_id, 0.0), reverse=True)
        ordered = ordered[: max(top_k, pool_size // 2)]

        results = [self._to_result(c, fused.get(c.chunk_id)) for c in ordered]

        if q.use_reranker and self._reranker is not None and len(results) > 1:
            results = await self._rerank(q.query, results)

        return results[:top_k]

    async def _dense(
        self, q: RetrievalQuery, embedding: list[float], pool: int
    ) -> list[_Candidate]:
        where, args = _build_filter_clause(q, start_index=2)
        sql = f"""
        SELECT {_BASE_SELECT}, 1 - (c.embedding <=> $1::vector) AS dense_score
        FROM chunks c
        JOIN filings f ON f.id = c.filing_id
        JOIN filing_sections s ON s.id = c.section_id
        {where}
        ORDER BY c.embedding <=> $1::vector ASC
        LIMIT {pool}
        """
        async with self._conn() as conn:
            rows = await conn.fetch(sql, _format_vector(embedding), *args)
        return [
            _Candidate(
                chunk_id=r["chunk_id"],
                filing_id=r["filing_id"],
                section_id=r["section_id"],
                item_code=r["item_code"],
                text=r["text"],
                ticker=r["ticker"],
                accession=r["accession"],
                form_type=r["form_type"],
                filed_at=r["filed_at"],
                source_url=r["source_url"],
                chunk_ord=r["chunk_ord"],
                dense_score=float(r["dense_score"]),
            )
            for r in rows
        ]

    async def _lexical(self, q: RetrievalQuery, pool: int) -> list[_Candidate]:
        where, args = _build_filter_clause(q, start_index=2)
        sql = f"""
        SELECT {_BASE_SELECT},
               ts_rank_cd(to_tsvector('english', c.text), plainto_tsquery('english', $1))
                  AS lexical_score
        FROM chunks c
        JOIN filings f ON f.id = c.filing_id
        JOIN filing_sections s ON s.id = c.section_id
        WHERE to_tsvector('english', c.text) @@ plainto_tsquery('english', $1)
          {("AND " + where[len("WHERE ") :]) if where else ""}
        ORDER BY lexical_score DESC
        LIMIT {pool}
        """
        async with self._conn() as conn:
            rows = await conn.fetch(sql, q.query, *args)
        return [
            _Candidate(
                chunk_id=r["chunk_id"],
                filing_id=r["filing_id"],
                section_id=r["section_id"],
                item_code=r["item_code"],
                text=r["text"],
                ticker=r["ticker"],
                accession=r["accession"],
                form_type=r["form_type"],
                filed_at=r["filed_at"],
                source_url=r["source_url"],
                chunk_ord=r["chunk_ord"],
                lexical_score=float(r["lexical_score"]),
            )
            for r in rows
        ]

    @staticmethod
    def _merge(dense: list[_Candidate], lexical: list[_Candidate]) -> dict[int, _Candidate]:
        out: dict[int, _Candidate] = {}
        for c in dense:
            out[c.chunk_id] = c
        for c in lexical:
            existing = out.get(c.chunk_id)
            if existing is None:
                out[c.chunk_id] = c
            else:
                existing.lexical_score = c.lexical_score
        return out

    def _to_result(self, c: _Candidate, fused: float | None) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=c.chunk_id,
            filing_id=c.filing_id,
            section_id=c.section_id,
            item_code=c.item_code,
            text=c.text,
            ticker=c.ticker,
            accession=c.accession,
            form_type=c.form_type,
            filed_at=c.filed_at,
            source_url=c.source_url,
            chunk_ord=c.chunk_ord,
            dense_score=c.dense_score,
            lexical_score=c.lexical_score,
            fused_score=fused,
        )

    async def _rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        assert self._reranker is not None
        ranked = await self._reranker.rerank(query, [r.text for r in results])
        out: list[RetrievalResult] = []
        for r in ranked:
            res = results[r.index]
            res.rerank_score = r.score
            out.append(res)
        return out


@dataclass(slots=True)
class _FilterBuild:
    where: str
    args: list[object] = field(default_factory=list)


def _build_filter_clause(q: RetrievalQuery, *, start_index: int) -> tuple[str, list[object]]:
    """Build a parameterized WHERE clause + arg list, starting placeholders at $start_index."""
    clauses: list[str] = []
    args: list[object] = []
    idx = start_index

    if q.ticker:
        clauses.append(f"f.ticker = ${idx}")
        args.append(q.ticker.upper())
        idx += 1
    if q.form_types:
        clauses.append(f"f.form_type = ANY(${idx}::text[])")
        args.append(q.form_types)
        idx += 1
    if q.item_codes:
        clauses.append(f"s.item_code = ANY(${idx}::text[])")
        args.append(q.item_codes)
        idx += 1
    if q.filed_after:
        clauses.append(f"f.filed_at >= ${idx}::timestamptz")
        args.append(q.filed_after)
        idx += 1

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, args
