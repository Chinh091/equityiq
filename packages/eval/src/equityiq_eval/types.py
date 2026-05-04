from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    """One row in golden Q/A dataset."""

    id: str
    question: str
    reference_answer: str
    expected_accessions: list[str] = Field(default_factory=list)
    expected_item_codes: list[str] = Field(default_factory=list)
    ticker: str | None = None
    tags: list[str] = Field(default_factory=list)


class JudgeScore(BaseModel):
    metric: str
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""


class EvalRowResult(BaseModel):
    item_id: str
    question: str
    answer: str
    retrieved_accessions: list[str]
    retrieved_item_codes: list[str]
    scores: list[JudgeScore]
    latency_s: float
    error: str | None = None

    def score_for(self, metric: str) -> float | None:
        for s in self.scores:
            if s.metric == metric:
                return s.score
        return None


class MetricSummary(BaseModel):
    metric: str
    mean: float
    n: int


class EvalReport(BaseModel):
    """Aggregate report written as JSON. Compared against baseline by ci_gate."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    dataset: str
    n_items: int
    n_errors: int
    metrics: list[MetricSummary]
    rows: list[EvalRowResult]

    def metric_mean(self, metric: str) -> float | None:
        for m in self.metrics:
            if m.metric == metric:
                return m.mean
        return None
