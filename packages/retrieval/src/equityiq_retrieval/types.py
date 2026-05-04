from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """A retrieved chunk plus the metadata needed to cite it.

    Scores: dense (cosine), lexical (ts_rank_cd), fused (RRF), rerank (cross-encoder).
    Whichever stage hasn't run is None; the API returns the most-trusted score
    available (rerank > fused > dense).
    """

    chunk_id: int
    filing_id: int
    section_id: int
    item_code: str
    text: str
    ticker: str | None
    accession: str
    form_type: str
    filed_at: datetime
    source_url: str
    chunk_ord: int

    dense_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    @property
    def best_score(self) -> float:
        for s in (self.rerank_score, self.fused_score, self.dense_score, self.lexical_score):
            if s is not None:
                return s
        return 0.0
