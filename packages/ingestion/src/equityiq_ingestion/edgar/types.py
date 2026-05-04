from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class FormType(StrEnum):
    K10 = "10-K"
    Q10 = "10-Q"
    K8 = "8-K"
    F20 = "20-F"


class FilingMeta(BaseModel):
    """Subset of EDGAR submission metadata we persist."""

    accession: str = Field(..., description="0000320193-24-000123 style accession number")
    cik: str
    ticker: str | None = None
    form_type: FormType
    filed_at: datetime
    period_of_report: date | None = None
    primary_document: str
    source_url: str

    @property
    def normalized_accession(self) -> str:
        return self.accession.replace("-", "")
