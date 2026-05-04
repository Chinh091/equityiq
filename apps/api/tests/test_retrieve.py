from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from equityiq_api import app
from equityiq_api.deps import get_retriever
from equityiq_retrieval import RetrievalResult


@pytest.fixture
def fake_retriever() -> AsyncMock:
    fake = AsyncMock()
    fake.search.return_value = [
        RetrievalResult(
            chunk_id=1,
            filing_id=10,
            section_id=100,
            item_code="1A",
            text="Risk: supply chain dependence on TSMC.",
            ticker="NVDA",
            accession="0001045810-24-000001",
            form_type="10-K",
            filed_at=datetime(2024, 2, 21),
            source_url="https://example.test/doc.htm",
            chunk_ord=0,
            dense_score=0.81,
            lexical_score=0.42,
            fused_score=0.91,
            rerank_score=0.95,
        )
    ]
    return fake


@pytest.fixture
def http(fake_retriever: AsyncMock):
    app.dependency_overrides[get_retriever] = lambda: fake_retriever
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_retriever, None)


def test_retrieve_returns_results(http: TestClient) -> None:
    resp = http.post(
        "/retrieve",
        json={"query": "supply chain risk", "ticker": "NVDA", "top_k": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["used_reranker"] is True
    assert body["results"][0]["item_code"] == "1A"
    assert body["results"][0]["rerank_score"] == 0.95


def test_retrieve_validates_query_min_len(http: TestClient) -> None:
    resp = http.post("/retrieve", json={"query": "x"})
    assert resp.status_code == 422
