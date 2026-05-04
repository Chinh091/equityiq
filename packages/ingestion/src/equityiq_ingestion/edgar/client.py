from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from equityiq_ingestion.config import IngestionSettings
from equityiq_ingestion.edgar.types import FilingMeta, FormType


class EdgarError(RuntimeError):
    pass


class RateLimit:
    """Token-bucket-ish rate limiter (cheap; good enough for a single client)."""

    def __init__(self, max_rps: float) -> None:
        self._min_interval = 1.0 / max_rps
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._last + self._min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class EdgarClient:
    """Async EDGAR client.

    Endpoints used:
      - data.sec.gov/submissions/CIK{padded}.json  -> filing index per company
      - www.sec.gov/cgi-bin/browse-edgar           -> ticker→CIK lookup (fallback)
      - www.sec.gov/Archives/edgar/data/...        -> filing documents

    Note: EDGAR rejects requests without a descriptive User-Agent containing a
    contact email. Set SEC_EDGAR_USER_AGENT in .env.
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or IngestionSettings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=self._settings.sec_edgar_timeout_s,
            headers={
                "User-Agent": self._settings.sec_edgar_user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov",
            },
        )
        self._limiter = RateLimit(self._settings.sec_edgar_max_rps)

    async def __aenter__(self) -> EdgarClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, max=30),
        reraise=True,
    )
    async def _get(self, url: str, *, host: str | None = None) -> httpx.Response:
        await self._limiter.acquire()
        headers = {"Host": host} if host else None
        resp = await self._client.get(url, headers=headers)
        if resp.status_code == 429:
            # Respect Retry-After if present, then retry via tenacity.
            retry_after = float(resp.headers.get("Retry-After", "2"))
            await asyncio.sleep(retry_after)
            resp.raise_for_status()
        resp.raise_for_status()
        return resp

    @staticmethod
    def _pad_cik(cik: str | int) -> str:
        return str(cik).lstrip("0").zfill(10)

    async def submissions(self, cik: str | int) -> dict[str, Any]:
        url = f"{self._settings.sec_edgar_data_url}/submissions/CIK{self._pad_cik(cik)}.json"
        resp = await self._get(url, host="data.sec.gov")
        return resp.json()  # type: ignore[no-any-return]

    async def list_filings(
        self,
        cik: str | int,
        *,
        form_types: Iterable[FormType] | None = None,
        limit: int | None = None,
    ) -> list[FilingMeta]:
        """Return recent filings for a CIK, optionally filtered by form type."""
        data = await self.submissions(cik)
        recent = data.get("filings", {}).get("recent", {})
        out = self._parse_recent(cik=str(cik), data=data, recent=recent, form_types=form_types)
        if limit is not None:
            out = out[:limit]
        return out

    @staticmethod
    def _parse_recent(
        *,
        cik: str,
        data: dict[str, Any],
        recent: dict[str, list[Any]],
        form_types: Iterable[FormType] | None,
    ) -> list[FilingMeta]:
        forms: Sequence[str] = recent.get("form", [])
        accs: Sequence[str] = recent.get("accessionNumber", [])
        filed: Sequence[str] = recent.get("filingDate", [])
        period: Sequence[str] = recent.get("reportDate", [])
        primary: Sequence[str] = recent.get("primaryDocument", [])

        wanted = {f.value for f in form_types} if form_types else None
        ticker_list = data.get("tickers", []) or [None]
        ticker = ticker_list[0]

        filings: list[FilingMeta] = []
        for i, form in enumerate(forms):
            if wanted and form not in wanted:
                continue
            try:
                form_type = FormType(form)
            except ValueError:
                continue
            acc = accs[i]
            acc_nodash = acc.replace("-", "")
            doc = primary[i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
            filed_at = datetime.fromisoformat(filed[i])
            period_of_report = None
            p = period[i] if i < len(period) else ""
            if p:
                from datetime import date

                period_of_report = date.fromisoformat(p)
            filings.append(
                FilingMeta(
                    accession=acc,
                    cik=str(int(cik)),
                    ticker=ticker,
                    form_type=form_type,
                    filed_at=filed_at,
                    period_of_report=period_of_report,
                    primary_document=doc,
                    source_url=url,
                )
            )
        return filings

    async def fetch_document(self, filing: FilingMeta) -> str:
        resp = await self._get(filing.source_url, host="www.sec.gov")
        return resp.text

    async def lookup_cik(self, ticker: str) -> str:
        """Resolve ticker → zero-padded CIK using the company_tickers.json file."""
        url = f"{self._settings.sec_edgar_base_url}/files/company_tickers.json"
        resp = await self._get(url, host="www.sec.gov")
        for entry in resp.json().values():
            if str(entry.get("ticker", "")).upper() == ticker.upper():
                return self._pad_cik(entry["cik_str"])
        raise EdgarError(f"ticker not found in company_tickers.json: {ticker}")
