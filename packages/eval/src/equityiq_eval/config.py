from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    eval_concurrency: int = 4
    eval_top_k: int = 10
    eval_judge_temperature: float = 0.0
    eval_max_regression: float = 0.03
    eval_report_path: str = "eval-report.json"
    eval_baseline_path: str = "eval-baseline.json"
