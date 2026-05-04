import httpx
import pytest
import respx

from equityiq_ingestion import EdgarClient, FormType
from equityiq_ingestion.config import IngestionSettings


@pytest.fixture
def settings() -> IngestionSettings:
    return IngestionSettings(
        sec_edgar_user_agent="EquityIQ test you@example.com",
        sec_edgar_max_rps=1000.0,  # don't actually wait in tests
    )


@pytest.mark.asyncio
@respx.mock
async def test_list_filings_filters_by_form_type(settings: IngestionSettings) -> None:
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "tickers": ["AAPL"],
                "filings": {
                    "recent": {
                        "form": ["10-K", "8-K", "10-Q", "DEF 14A"],
                        "accessionNumber": [
                            "0000320193-24-000123",
                            "0000320193-24-000124",
                            "0000320193-24-000125",
                            "0000320193-24-000126",
                        ],
                        "filingDate": [
                            "2024-11-01",
                            "2024-10-15",
                            "2024-08-01",
                            "2024-09-01",
                        ],
                        "reportDate": ["2024-09-28", "", "2024-06-29", ""],
                        "primaryDocument": [
                            "aapl-20240928.htm",
                            "aapl-8k.htm",
                            "aapl-10q.htm",
                            "aapl-def14a.htm",
                        ],
                    }
                },
            },
        )
    )

    async with EdgarClient(settings) as c:
        filings = await c.list_filings("320193", form_types=[FormType.K10, FormType.Q10])

    assert {f.form_type for f in filings} == {FormType.K10, FormType.Q10}
    assert len(filings) == 2
    assert filings[0].accession == "0000320193-24-000123"
    assert "aapl-20240928.htm" in filings[0].source_url


@pytest.mark.asyncio
@respx.mock
async def test_lookup_cik_resolves_ticker(settings: IngestionSettings) -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                "1": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com Inc."},
            },
        )
    )
    async with EdgarClient(settings) as c:
        cik = await c.lookup_cik("aapl")
    assert cik == "0000320193"


@pytest.mark.asyncio
@respx.mock
async def test_429_triggers_retry(settings: IngestionSettings) -> None:
    route = respx.get("https://data.sec.gov/submissions/CIK0000000001.json").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"tickers": [], "filings": {"recent": {}}}),
        ]
    )
    async with EdgarClient(settings) as c:
        await c.submissions(1)
    assert route.call_count == 2
