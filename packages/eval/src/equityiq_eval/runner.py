from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from statistics import mean

from equityiq_llm import LLMClient
from equityiq_observability import get_logger, observed
from equityiq_retrieval import HybridRetriever
from equityiq_retrieval.hybrid import RetrievalQuery
from equityiq_retrieval.types import RetrievalResult

from equityiq_eval.config import EvalSettings
from equityiq_eval.judge import LLMJudge, answer_relevance, context_precision, faithfulness
from equityiq_eval.types import (
    EvalReport,
    EvalRowResult,
    GoldenItem,
    JudgeScore,
    MetricSummary,
)

log = get_logger(__name__)


AnswerFn = Callable[[str, list[RetrievalResult]], Awaitable[str]]


def load_dataset(path: str | Path) -> list[GoldenItem]:
    p = Path(path)
    items: list[GoldenItem] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no} bad JSON: {e}") from e
            items.append(GoldenItem.model_validate(obj))
    return items


async def _default_answer_fn(question: str, contexts: list[RetrievalResult]) -> str:
    """Stub answer generator for retrieval-only eval. Concatenates top contexts."""
    if not contexts:
        return "(no contexts retrieved)"
    snippets = [c.text[:300] for c in contexts[:3]]
    return f"Based on retrieved filings: {' '.join(snippets)}"


class EvalRunner:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        llm: LLMClient,
        settings: EvalSettings | None = None,
        answer_fn: AnswerFn | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings or EvalSettings()
        self._answer_fn = answer_fn or _default_answer_fn
        self._judge = LLMJudge(llm, temperature=self._settings.eval_judge_temperature)

    @observed("eval.run")
    async def run(self, dataset_path: str | Path) -> EvalReport:
        items = load_dataset(dataset_path)
        sem = asyncio.Semaphore(self._settings.eval_concurrency)

        async def _bound(item: GoldenItem) -> EvalRowResult:
            async with sem:
                return await self._run_one(item)

        rows = await asyncio.gather(*(_bound(it) for it in items))
        return self._aggregate(str(dataset_path), rows)

    async def _run_one(self, item: GoldenItem) -> EvalRowResult:
        t0 = time.perf_counter()
        try:
            results = await self._retriever.search(
                RetrievalQuery(
                    query=item.question,
                    ticker=item.ticker,
                    top_k=self._settings.eval_top_k,
                )
            )
            answer = await self._answer_fn(item.question, results)
            scores = await self._score(item, answer, results)
            return EvalRowResult(
                item_id=item.id,
                question=item.question,
                answer=answer,
                retrieved_accessions=[r.accession for r in results],
                retrieved_item_codes=[r.item_code for r in results],
                scores=scores,
                latency_s=time.perf_counter() - t0,
            )
        except Exception as e:
            log.warning("eval_row_failed", item_id=item.id, error=str(e))
            return EvalRowResult(
                item_id=item.id,
                question=item.question,
                answer="",
                retrieved_accessions=[],
                retrieved_item_codes=[],
                scores=[],
                latency_s=time.perf_counter() - t0,
                error=str(e),
            )

    async def _score(
        self, item: GoldenItem, answer: str, results: list[RetrievalResult]
    ) -> list[JudgeScore]:
        contexts = [r.text for r in results]
        retrieved_accessions = [r.accession for r in results]
        faith_task = faithfulness(
            self._judge, question=item.question, answer=answer, contexts=contexts
        )
        rel_task = answer_relevance(
            self._judge, question=item.question, answer=answer, reference=item.reference_answer
        )
        precision = context_precision(
            retrieved_accessions=retrieved_accessions,
            expected_accessions=item.expected_accessions,
        )
        faith, rel = await asyncio.gather(faith_task, rel_task)
        return [faith, rel, precision]

    @staticmethod
    def _aggregate(dataset: str, rows: list[EvalRowResult]) -> EvalReport:
        metric_names: list[str] = []
        for r in rows:
            for s in r.scores:
                if s.metric not in metric_names:
                    metric_names.append(s.metric)

        metrics: list[MetricSummary] = []
        for m in metric_names:
            vals = [s.score for r in rows for s in r.scores if s.metric == m]
            if vals:
                metrics.append(MetricSummary(metric=m, mean=mean(vals), n=len(vals)))

        return EvalReport(
            dataset=dataset,
            n_items=len(rows),
            n_errors=sum(1 for r in rows if r.error is not None),
            metrics=metrics,
            rows=rows,
        )


def write_report(report: EvalReport, path: str | Path) -> None:
    Path(path).write_text(report.model_dump_json(indent=2), encoding="utf-8")
