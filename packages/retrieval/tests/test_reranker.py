import httpx
import pytest
import respx
from equityiq_retrieval import RetrievalSettings, TEIReranker


@pytest.fixture
def settings() -> RetrievalSettings:
    return RetrievalSettings(reranker_base_url="http://tei.test")


@pytest.mark.asyncio
@respx.mock
async def test_rerank_sorts_by_score_descending(settings: RetrievalSettings) -> None:
    respx.post("http://tei.test/rerank").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"index": 0, "score": 0.10},
                {"index": 1, "score": 0.91},
                {"index": 2, "score": 0.55},
            ],
        )
    )
    async with TEIReranker(settings) as r:
        out = await r.rerank("query", ["a", "b", "c"])
    assert [x.index for x in out] == [1, 2, 0]
    assert out[0].score == 0.91


@pytest.mark.asyncio
@respx.mock
async def test_rerank_top_k_truncates(settings: RetrievalSettings) -> None:
    respx.post("http://tei.test/rerank").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"index": 0, "score": 0.1},
                {"index": 1, "score": 0.9},
                {"index": 2, "score": 0.5},
            ],
        )
    )
    async with TEIReranker(settings) as r:
        out = await r.rerank("q", ["a", "b", "c"], top_k=2)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_rerank_empty_returns_empty(settings: RetrievalSettings) -> None:
    async with TEIReranker(settings) as r:
        out = await r.rerank("q", [])
    assert out == []
