"""EquityIQ CLI.

Commands:
    ingest   — pull EDGAR filings, parse, chunk, embed, persist.
    query    — hybrid retrieve + rerank, print top-k chunks with citations.
    health   — sanity-check Ollama / Postgres / TEI connectivity.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from equityiq_ingestion import (
    Database,
    EdgarClient,
    FormType,
    IngestionPipeline,
    IngestStats,
)
from equityiq_llm import OllamaClient
from equityiq_observability import configure_logging, shutdown_langfuse
from equityiq_retrieval import (
    HybridRetriever,
    RetrievalQuery,
    RetrievalSettings,
    TEIReranker,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="EquityIQ CLI")
console = Console()


def _async(fn):
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


@app.callback()
def _root() -> None:
    configure_logging()


@app.command()
@_async
async def ingest(
    ticker: Annotated[str, typer.Option("--ticker", "-t", help="e.g. NVDA")],
    forms: Annotated[
        str, typer.Option("--forms", help="Comma list. Default: 10-K,10-Q")
    ] = "10-K,10-Q",
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max filings to ingest")] = 4,
) -> None:
    """Ingest filings for TICKER."""
    form_types = tuple(FormType(f.strip()) for f in forms.split(","))

    db = Database()
    await db.connect()
    try:
        async with EdgarClient() as edgar, OllamaClient() as llm:
            pipeline = IngestionPipeline(edgar=edgar, llm=llm, db=db)
            stats: IngestStats = await pipeline.run(
                ticker=ticker.upper(), forms=form_types, limit=limit
            )
        _print_stats(stats)
    finally:
        await db.close()
        shutdown_langfuse()


@app.command()
@_async
async def query(
    question: Annotated[str, typer.Argument(help="Natural-language question")],
    ticker: Annotated[str | None, typer.Option("--ticker", "-t")] = None,
    top_k: Annotated[int, typer.Option("--k", "-k")] = 8,
    no_rerank: Annotated[bool, typer.Option("--no-rerank", help="Skip cross-encoder rerank")] = False,
    item: Annotated[
        str | None, typer.Option("--item", help="Restrict to item code, e.g. 1A,7")
    ] = None,
) -> None:
    """Hybrid retrieve + rerank for QUESTION."""
    settings = RetrievalSettings()
    item_codes = [c.strip() for c in item.split(",")] if item else None

    async with OllamaClient() as llm, TEIReranker(settings) as reranker:
        retriever = HybridRetriever(llm=llm, reranker=reranker, settings=settings)
        try:
            await retriever.connect()
            results = await retriever.search(
                RetrievalQuery(
                    query=question,
                    ticker=ticker,
                    item_codes=item_codes,
                    top_k=top_k,
                    use_reranker=not no_rerank,
                )
            )
        finally:
            await retriever.aclose()

    _print_results(results)
    shutdown_langfuse()


@app.command()
@_async
async def health() -> None:
    """Smoke-check Ollama generate + DB connect + reranker."""
    ok: dict[str, str] = {}
    try:
        async with OllamaClient() as c:
            txt = await c.generate(prompt="say ok in one word")
            ok["ollama"] = f"ok: {txt[:32]}"
    except Exception as e:
        ok["ollama"] = f"FAIL: {e}"

    db = Database()
    try:
        await db.connect()
        ok["postgres"] = "ok"
    except Exception as e:
        ok["postgres"] = f"FAIL: {e}"
    finally:
        await db.close()

    settings = RetrievalSettings()
    try:
        async with TEIReranker(settings) as r:
            await r.rerank("hi", ["world"])
            ok["reranker"] = "ok"
    except Exception as e:
        ok["reranker"] = f"FAIL: {e}"

    table = Table(title="EquityIQ health")
    table.add_column("component")
    table.add_column("status")
    for k, v in ok.items():
        table.add_row(k, v)
    console.print(table)


def _print_stats(stats: IngestStats) -> None:
    t = Table(title="Ingestion summary")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("filings_processed", str(stats.filings_processed))
    t.add_row("sections_persisted", str(stats.sections_persisted))
    t.add_row("chunks_persisted", str(stats.chunks_persisted))
    t.add_row("skipped_no_sections", str(stats.skipped_no_sections))
    t.add_row("errors", str(len(stats.errors)))
    console.print(t)
    if stats.errors:
        console.print("[yellow]first errors:[/yellow]")
        for e in stats.errors[:5]:
            console.print(f"  - {e}")


def _print_results(results) -> None:
    if not results:
        console.print("[yellow]no results[/yellow]")
        return
    for i, r in enumerate(results, start=1):
        head = (
            f"[bold]{i}.[/bold] {r.ticker or '?'} {r.form_type} "
            f"item {r.item_code} chunk #{r.chunk_ord}  "
            f"score={r.best_score:.4f}"
        )
        console.rule(head)
        console.print(r.text[:1200])
        console.print(f"[dim]source: {r.source_url}[/dim]")
