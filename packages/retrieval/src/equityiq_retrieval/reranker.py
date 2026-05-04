"""text-embeddings-inference cross-encoder reranker client.

TEI's /rerank endpoint takes {query, texts} and returns scores per text.
We pass full chunk text; reranker handles its own truncation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from equityiq_retrieval.config import RetrievalSettings


class RerankerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    score: float


class TEIReranker:
    def __init__(
        self,
        settings: RetrievalSettings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or RetrievalSettings()
        self._owns = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._settings.reranker_base_url,
            timeout=self._settings.reranker_timeout_s,
        )

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()

    async def __aenter__(self) -> TEIReranker:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def rerank(
        self,
        query: str,
        texts: Sequence[str],
        *,
        top_k: int | None = None,
        truncate: bool = True,
    ) -> list[RerankResult]:
        if not texts:
            return []
        body = {
            "query": query,
            "texts": list(texts),
            "truncate": truncate,
            "raw_scores": False,
        }
        resp = await self._client.post("/rerank", json=body)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RerankerError(f"unexpected rerank shape: {data!r}")
        out = [RerankResult(index=int(r["index"]), score=float(r["score"])) for r in data]
        out.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            out = out[:top_k]
        return out
