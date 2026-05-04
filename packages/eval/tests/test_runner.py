from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from equityiq_eval.config import EvalSettings
from equityiq_eval.runner import EvalRunner, load_dataset, write_report
from equityiq_eval.types import EvalReport, GoldenItem
from equityiq_retrieval.types import RetrievalResult


def _gold(id_: str = "g1", **kw: object) -> GoldenItem:
    base: dict[str, object] = {
        "id": id_,
        "question": "what?",
        "reference_answer": "ref",
        "expected_accessions": ["acc-1"],
    }
    base.update(kw)
    return GoldenItem.model_validate(base)


def _retrieval(accession: str = "acc-1", text: str = "ctx") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=1,
        filing_id=1,
        section_id=1,
        item_code="1A",
        text=text,
        ticker="AAPL",
        accession=accession,
        form_type="10-K",
        filed_at=datetime(2024, 1, 1),
        source_url="https://example.test/f.htm",
        chunk_ord=0,
        rerank_score=0.9,
    )


def test_load_dataset_skips_blanks_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n"
        "# header comment\n"
        '{"id":"a","question":"q","reference_answer":"r"}\n'
        "\n"
        '{"id":"b","question":"q2","reference_answer":"r2"}\n',
        encoding="utf-8",
    )
    items = load_dataset(p)
    assert [i.id for i in items] == ["a", "b"]


def test_load_dataset_raises_on_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("not-json-at-all\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad JSON"):
        load_dataset(p)


@pytest.mark.asyncio
async def test_runner_aggregates_metrics(tmp_path: Path) -> None:
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        json.dumps(_gold("g1").model_dump(mode="json"))
        + "\n"
        + json.dumps(_gold("g2", expected_accessions=["acc-2"]).model_dump(mode="json"))
        + "\n",
        encoding="utf-8",
    )

    retriever = AsyncMock()
    retriever.search = AsyncMock(return_value=[_retrieval("acc-1"), _retrieval("acc-3")])

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"score": 0.8, "rationale": "ok"}')

    runner = EvalRunner(
        retriever=retriever,
        llm=llm,
        settings=EvalSettings(eval_concurrency=2),
        answer_fn=AsyncMock(return_value="generated answer"),
    )
    report = await runner.run(ds)

    assert report.n_items == 2
    assert report.n_errors == 0
    metrics = {m.metric: m for m in report.metrics}
    assert "faithfulness" in metrics
    assert "answer_relevance" in metrics
    assert "context_precision" in metrics
    # g1 expected acc-1, retrieved [acc-1, acc-3] → 1/2; g2 expected acc-2 → 0/2.
    assert metrics["context_precision"].mean == pytest.approx(0.25)
    assert metrics["faithfulness"].mean == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_runner_records_row_errors(tmp_path: Path) -> None:
    ds = tmp_path / "ds.jsonl"
    ds.write_text(json.dumps(_gold("g1").model_dump(mode="json")) + "\n", encoding="utf-8")

    retriever = AsyncMock()
    retriever.search = AsyncMock(side_effect=RuntimeError("boom"))

    llm = AsyncMock()
    runner = EvalRunner(retriever=retriever, llm=llm, answer_fn=AsyncMock(return_value="x"))
    report = await runner.run(ds)

    assert report.n_errors == 1
    assert report.rows[0].error == "boom"
    assert report.rows[0].scores == []


def test_write_report_round_trips(tmp_path: Path) -> None:
    report = EvalReport(dataset="ds", n_items=0, n_errors=0, metrics=[], rows=[])
    out = tmp_path / "r.json"
    write_report(report, out)
    loaded = EvalReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert loaded.dataset == "ds"
