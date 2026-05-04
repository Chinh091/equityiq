from __future__ import annotations

from dataclasses import dataclass

from equityiq_retrieval import HybridRetriever
from equityiq_retrieval.hybrid import RetrievalQuery
from equityiq_retrieval.types import RetrievalResult


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str


class RetrieveTool:
    spec = ToolSpec(
        name="retrieve_filings",
        description="Search SEC filings for chunks relevant to a question. Args: query, ticker?, top_k?",
    )

    def __init__(self, retriever: HybridRetriever, *, default_top_k: int = 8) -> None:
        self._retriever = retriever
        self._default_top_k = default_top_k

    async def __call__(
        self, *, query: str, ticker: str | None = None, top_k: int | None = None
    ) -> list[RetrievalResult]:
        return await self._retriever.search(
            RetrievalQuery(query=query, ticker=ticker, top_k=top_k or self._default_top_k)
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, object] = {}

    def register(self, name: str, tool: object) -> None:
        self._tools[name] = tool

    def get(self, name: str) -> object:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values() if hasattr(t, "spec")]
