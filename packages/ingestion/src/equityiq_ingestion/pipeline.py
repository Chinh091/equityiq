"""Ingestion pipeline: EDGAR → parse → chunk → embed → persist.

Designed for both batch (CLI / cron) and one-shot (test fixtures) use.
Embedding is batched to avoid 1k-RTT latency hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from equityiq_ingestion.chunker import Chunk, SemanticChunker
from equityiq_ingestion.config import IngestionSettings
from equityiq_ingestion.db import Database
from equityiq_ingestion.edgar.client import EdgarClient
from equityiq_ingestion.edgar.types import FilingMeta, FormType
from equityiq_ingestion.parsers.sec_sections import parse_10k_html
from equityiq_llm import OllamaClient
from equityiq_observability import get_logger, observed

log = get_logger(__name__)


@dataclass(slots=True)
class IngestStats:
    filings_processed: int = 0
    sections_persisted: int = 0
    chunks_persisted: int = 0
    skipped_no_sections: int = 0
    errors: list[str] = field(default_factory=list)


class IngestionPipeline:
    """Composes EDGAR client, parser, chunker, embedder, DB.

    Caller owns lifecycle: instantiate dependencies, pass them in. Pipeline
    doesn't open/close them so it composes cleanly into CLI vs API contexts.
    """

    def __init__(
        self,
        *,
        edgar: EdgarClient,
        llm: OllamaClient,
        db: Database,
        chunker: SemanticChunker | None = None,
        settings: IngestionSettings | None = None,
        embed_batch_size: int = 32,
    ) -> None:
        s = settings or IngestionSettings()
        self._edgar = edgar
        self._llm = llm
        self._db = db
        self._chunker = chunker or SemanticChunker(
            target_tokens=s.chunk_target_tokens,
            overlap_tokens=s.chunk_overlap_tokens,
            max_tokens=s.chunk_max_tokens,
        )
        self._embed_batch = embed_batch_size

    @observed("ingest.run")
    async def run(
        self,
        *,
        ticker: str,
        forms: tuple[FormType, ...] = (FormType.K10, FormType.Q10),
        limit: int = 8,
    ) -> IngestStats:
        stats = IngestStats()
        cik = await self._edgar.lookup_cik(ticker)
        log.info("ingest.start", ticker=ticker, cik=cik, forms=[f.value for f in forms], limit=limit)
        filings = await self._edgar.list_filings(cik, form_types=forms, limit=limit)
        for f in filings:
            try:
                await self._ingest_one(f, stats)
            except Exception as e:  # one bad filing shouldn't break the batch
                log.warning("ingest.filing_failed", accession=f.accession, error=str(e))
                stats.errors.append(f"{f.accession}: {e}")
        log.info("ingest.done", **stats.__dict__)
        return stats

    @observed("ingest.filing")
    async def _ingest_one(self, meta: FilingMeta, stats: IngestStats) -> None:
        html = await self._edgar.fetch_document(meta)
        parsed = parse_10k_html(html)
        if not parsed.sections:
            stats.skipped_no_sections += 1
            log.info("ingest.skip_no_sections", accession=meta.accession)
            return

        filing = await self._db.upsert_filing(meta)
        section_ids = await self._db.replace_sections(filing.id, parsed.sections)

        for section in parsed.sections:
            section_id = section_ids[section.item_code]
            chunks = self._chunker.chunk(section.text)
            if not chunks:
                continue
            embeddings = await self._embed_chunks(chunks)
            await self._db.insert_chunks(
                filing_id=filing.id,
                section_id=section_id,
                chunks=chunks,
                embeddings=embeddings,
            )
            stats.sections_persisted += 1
            stats.chunks_persisted += len(chunks)

        stats.filings_processed += 1

    async def _embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(chunks), self._embed_batch):
            batch = chunks[i : i + self._embed_batch]
            vecs = await self._llm.embed([c.text for c in batch])
            out.extend(vecs)
        return out
