from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from equityiq_retrieval import RetrievalResult

Ticker = Annotated[str, StringConstraints(min_length=1, max_length=10, pattern=r"^[A-Z.\-]+$")]


class HealthResponse(BaseModel):
    status: str


class ThesisRequest(BaseModel):
    ticker: Ticker
    question: Annotated[str, Field(min_length=3, max_length=2000)]


class RetrieveRequest(BaseModel):
    query: Annotated[str, Field(min_length=2, max_length=2000)]
    ticker: Ticker | None = None
    form_types: list[str] | None = None
    item_codes: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    pool: int | None = Field(default=None, ge=1, le=200)
    use_reranker: bool = True


class RetrieveResponse(BaseModel):
    results: list[RetrievalResult]
    count: int
    used_reranker: bool
