from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from equityiq_cli.main import app
from equityiq_ingestion import IngestStats
from equityiq_retrieval.types import RetrievalResult


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _retrieval(text: str = "Apple discloses supply chain risks.") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=1,
        filing_id=10,
        section_id=100,
        item_code="1A",
        text=text,
        ticker="AAPL",
        accession="0000320193-24-000001",
        form_type="10-K",
        filed_at=datetime(2024, 1, 1),
        source_url="https://example.test/f.htm",
        chunk_ord=0,
        rerank_score=0.95,
    )


def _ctx(mock_obj: MagicMock) -> MagicMock:
    """Build an async context manager that yields mock_obj."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_obj)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def test_help_lists_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "query" in result.stdout
    assert "health" in result.stdout


def test_ingest_command_runs_pipeline(runner: CliRunner) -> None:
    stats = IngestStats(
        filings_processed=2,
        sections_persisted=8,
        chunks_persisted=42,
        skipped_no_sections=0,
        errors=[],
    )
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=stats)

    db = MagicMock()
    db.connect = AsyncMock()
    db.close = AsyncMock()

    with (
        patch("equityiq_cli.main.Database", return_value=db),
        patch("equityiq_cli.main.EdgarClient", return_value=_ctx(MagicMock())),
        patch("equityiq_cli.main.OllamaClient", return_value=_ctx(MagicMock())),
        patch("equityiq_cli.main.IngestionPipeline", return_value=pipeline),
    ):
        result = runner.invoke(app, ["ingest", "--ticker", "AAPL", "--limit", "2"])

    assert result.exit_code == 0, result.stdout
    pipeline.run.assert_awaited_once()
    kwargs = pipeline.run.await_args.kwargs
    assert kwargs["ticker"] == "AAPL"
    assert kwargs["limit"] == 2
    assert "filings_processed" in result.stdout
    assert "42" in result.stdout  # chunks count rendered


def test_query_command_prints_results(runner: CliRunner) -> None:
    retriever = MagicMock()
    retriever.connect = AsyncMock()
    retriever.aclose = AsyncMock()
    retriever.search = AsyncMock(return_value=[_retrieval()])

    with (
        patch("equityiq_cli.main.OllamaClient", return_value=_ctx(MagicMock())),
        patch("equityiq_cli.main.TEIReranker", return_value=_ctx(MagicMock())),
        patch("equityiq_cli.main.HybridRetriever", return_value=retriever),
    ):
        result = runner.invoke(app, ["query", "supply risk?", "--ticker", "AAPL", "--k", "3"])

    assert result.exit_code == 0, result.stdout
    retriever.search.assert_awaited_once()
    q = retriever.search.await_args.args[0]
    assert q.query == "supply risk?"
    assert q.ticker == "AAPL"
    assert q.top_k == 3
    assert q.use_reranker is True
    assert "Apple discloses supply chain risks" in result.stdout


def test_query_command_no_rerank_flag(runner: CliRunner) -> None:
    retriever = MagicMock()
    retriever.connect = AsyncMock()
    retriever.aclose = AsyncMock()
    retriever.search = AsyncMock(return_value=[])

    with (
        patch("equityiq_cli.main.OllamaClient", return_value=_ctx(MagicMock())),
        patch("equityiq_cli.main.TEIReranker", return_value=_ctx(MagicMock())),
        patch("equityiq_cli.main.HybridRetriever", return_value=retriever),
    ):
        result = runner.invoke(app, ["query", "x?", "--no-rerank"])

    assert result.exit_code == 0
    q = retriever.search.await_args.args[0]
    assert q.use_reranker is False
    assert "no results" in result.stdout


def test_health_reports_per_component_status(runner: CliRunner) -> None:
    ollama = MagicMock()
    ollama.generate = AsyncMock(return_value="ok")

    db = MagicMock()
    db.connect = AsyncMock()
    db.close = AsyncMock()

    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[])

    with (
        patch("equityiq_cli.main.OllamaClient", return_value=_ctx(ollama)),
        patch("equityiq_cli.main.Database", return_value=db),
        patch("equityiq_cli.main.TEIReranker", return_value=_ctx(reranker)),
    ):
        result = runner.invoke(app, ["health"])

    assert result.exit_code == 0, result.stdout
    assert "ollama" in result.stdout
    assert "postgres" in result.stdout
    assert "reranker" in result.stdout


def test_health_reports_failures_without_crashing(runner: CliRunner) -> None:
    ollama = MagicMock()
    ollama.generate = AsyncMock(side_effect=RuntimeError("connection refused"))

    db = MagicMock()
    db.connect = AsyncMock(side_effect=RuntimeError("db down"))
    db.close = AsyncMock()

    reranker = MagicMock()
    reranker.rerank = AsyncMock(side_effect=RuntimeError("tei down"))

    with (
        patch("equityiq_cli.main.OllamaClient", return_value=_ctx(ollama)),
        patch("equityiq_cli.main.Database", return_value=db),
        patch("equityiq_cli.main.TEIReranker", return_value=_ctx(reranker)),
    ):
        result = runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "FAIL" in result.stdout
