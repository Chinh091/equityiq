from __future__ import annotations

from equityiq_eval.ci_gate import _diff_table
from equityiq_eval.types import EvalReport, MetricSummary


def _report(metrics: dict[str, float]) -> EvalReport:
    return EvalReport(
        dataset="ds",
        n_items=10,
        n_errors=0,
        metrics=[MetricSummary(metric=k, mean=v, n=10) for k, v in metrics.items()],
        rows=[],
    )


def test_diff_table_no_regression() -> None:
    base = _report({"faithfulness": 0.80, "answer_relevance": 0.75})
    cur = _report({"faithfulness": 0.82, "answer_relevance": 0.74})
    _, failed = _diff_table(cur, base, max_reg=0.03)
    assert failed is False  # 0.74 vs 0.75 is a minor dip, within threshold


def test_diff_table_regression_fails() -> None:
    base = _report({"faithfulness": 0.80})
    cur = _report({"faithfulness": 0.74})  # 0.06 drop > 0.03 threshold
    _, failed = _diff_table(cur, base, max_reg=0.03)
    assert failed is True


def test_diff_table_new_metric_doesnt_fail() -> None:
    base = _report({"faithfulness": 0.80})
    cur = _report({"faithfulness": 0.81, "answer_relevance": 0.7})
    _, failed = _diff_table(cur, base, max_reg=0.03)
    assert failed is False


def test_diff_table_threshold_boundary() -> None:
    base = _report({"faithfulness": 0.80})
    # Comfortably within threshold.
    cur = _report({"faithfulness": 0.78})
    _, failed = _diff_table(cur, base, max_reg=0.03)
    assert failed is False
    # Comfortably past threshold.
    cur = _report({"faithfulness": 0.75})
    _, failed = _diff_table(cur, base, max_reg=0.03)
    assert failed is True
