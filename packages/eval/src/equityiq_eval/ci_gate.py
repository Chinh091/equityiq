"""CI regression gate.

Compares this branch's eval report against a baseline (typically main branch's
last green report) and exits non-zero if any metric regresses by more than
--max-regression.

Usage in CI:
    uv run python -m equityiq_eval.ci_gate \\
        --dataset packages/eval/golden/qa_2024.jsonl \\
        --baseline-from-main \\
        --max-regression 0.03
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import typer
from equityiq_llm import OllamaClient
from equityiq_retrieval import HybridRetriever
from equityiq_retrieval.reranker import TEIReranker
from rich.console import Console
from rich.table import Table

from equityiq_eval.config import EvalSettings
from equityiq_eval.runner import EvalRunner, write_report
from equityiq_eval.types import EvalReport

app = typer.Typer(add_completion=False, help="Eval CI gate")
console = Console()


def _load_baseline(path: Path) -> EvalReport | None:
    if not path.exists():
        return None
    try:
        return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:
        console.print(f"[yellow]baseline parse failed: {e}; treating as missing[/yellow]")
        return None


def _fetch_baseline_from_main(report_filename: str) -> EvalReport | None:
    """Try to fetch the baseline report from origin/main via git show."""
    try:
        raw = subprocess.check_output(
            ["git", "show", f"origin/main:{report_filename}"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return EvalReport.model_validate_json(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        console.print(f"[yellow]no baseline on origin/main ({e}); skipping comparison[/yellow]")
        return None


def _diff_table(current: EvalReport, baseline: EvalReport, max_reg: float) -> tuple[Table, bool]:
    table = Table(title="Eval metric diff (current vs baseline)")
    table.add_column("metric")
    table.add_column("baseline", justify="right")
    table.add_column("current", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("status")

    failed = False
    for cur in current.metrics:
        base = baseline.metric_mean(cur.metric)
        if base is None:
            table.add_row(cur.metric, "—", f"{cur.mean:.3f}", "—", "[cyan]new[/cyan]")
            continue
        delta = cur.mean - base
        if delta < -max_reg:
            status = f"[red]REGRESS (>{max_reg:.2f})[/red]"
            failed = True
        elif delta < 0:
            status = "[yellow]minor dip[/yellow]"
        else:
            status = "[green]ok[/green]"
        table.add_row(cur.metric, f"{base:.3f}", f"{cur.mean:.3f}", f"{delta:+.3f}", status)

    return table, failed


async def _run(
    dataset: Path,
    settings: EvalSettings,
) -> EvalReport:
    llm = OllamaClient()
    reranker = TEIReranker()
    retriever = HybridRetriever(llm=llm, reranker=reranker)
    try:
        await retriever.connect()
        runner = EvalRunner(retriever=retriever, llm=llm, settings=settings)
        return await runner.run(dataset)
    finally:
        await retriever.aclose()
        await reranker.aclose()
        await llm.aclose()


@app.command()
def main(
    dataset: Path = typer.Option(..., "--dataset", exists=True, readable=True),
    baseline_from_main: bool = typer.Option(False, "--baseline-from-main"),
    baseline_path: Path | None = typer.Option(None, "--baseline-path"),
    max_regression: float = typer.Option(0.03, "--max-regression"),
    report_path: Path = typer.Option(Path("eval-report.json"), "--report-path"),
) -> None:
    settings = EvalSettings(eval_max_regression=max_regression)

    report = asyncio.run(_run(dataset, settings))
    write_report(report, report_path)

    console.print(
        f"[bold]Eval complete:[/bold] {report.n_items} items, "
        f"{report.n_errors} errors → {report_path}"
    )
    summary = Table(title="Current metrics")
    summary.add_column("metric")
    summary.add_column("mean", justify="right")
    summary.add_column("n", justify="right")
    for m in report.metrics:
        summary.add_row(m.metric, f"{m.mean:.3f}", str(m.n))
    console.print(summary)

    baseline: EvalReport | None = None
    if baseline_path is not None:
        baseline = _load_baseline(baseline_path)
    elif baseline_from_main:
        baseline = _fetch_baseline_from_main(str(report_path))

    if baseline is None:
        console.print("[cyan]no baseline available — skipping regression gate[/cyan]")
        raise typer.Exit(0)

    table, failed = _diff_table(report, baseline, max_regression)
    console.print(table)
    if failed:
        console.print("[bold red]CI gate FAILED: metric regression exceeds threshold[/bold red]")
        raise typer.Exit(1)
    console.print("[bold green]CI gate PASSED[/bold green]")


if __name__ == "__main__":
    app()
