"""Multi-agent orchestration loop.

Pipeline per query:
    1. Planner decomposes question into sub-queries (Primary tier).
    2. RetrieveTool runs each sub-query against HybridRetriever.
    3. Analyst synthesizes a draft answer with citations (Primary tier).
    4. Critic scores faithfulness vs contexts (Judge tier).
    5. If faithfulness < threshold and revision_rounds remaining, re-analyze
       with a "be more conservative" prefix and re-critique.
    6. Emit FinalAnswerEvent with deduped citation list.

All steps yield AgentEvent objects so callers can stream over SSE.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from equityiq_llm import LLMClient
from equityiq_observability import get_logger
from equityiq_retrieval.types import RetrievalResult

from equityiq_agents.config import AgentSettings
from equityiq_agents.events import (
    AgentEvent,
    CritiqueEvent,
    DraftEvent,
    FinalAnswerEvent,
    PlanEvent,
    ToolCallEvent,
    ToolResultEvent,
    serialize_event,
)
from equityiq_agents.roles import analyze, critique, plan
from equityiq_agents.tools import RetrieveTool

log = get_logger(__name__)


class AgentLoop:
    def __init__(
        self,
        *,
        llm: LLMClient,
        retrieve: RetrieveTool,
        settings: AgentSettings | None = None,
    ) -> None:
        self._llm = llm
        self._retrieve = retrieve
        self._settings = settings or AgentSettings()

    async def run(self, *, question: str, ticker: str | None = None) -> AsyncIterator[AgentEvent]:
        subqueries = await plan(
            self._llm,
            question=question,
            max_subqueries=self._settings.agent_max_subqueries,
        )
        yield PlanEvent(subqueries=subqueries)

        contexts: list[RetrievalResult] = []
        for sub in subqueries:
            yield ToolCallEvent(name="retrieve_filings", args={"query": sub, "ticker": ticker})
            results = await self._retrieve(
                query=sub, ticker=ticker, top_k=self._settings.agent_retrieve_top_k
            )
            contexts.extend(results)
            yield ToolResultEvent(
                name="retrieve_filings",
                summary=f"sub='{sub[:60]}' → {len(results)} chunks",
                n_results=len(results),
            )

        ctx_pairs = _dedupe_contexts(contexts)
        draft = await analyze(self._llm, question=question, contexts=ctx_pairs)
        yield DraftEvent(text=draft)

        score, notes = await critique(self._llm, draft=draft, contexts=ctx_pairs)
        accepted = score >= self._settings.agent_critic_min_faithfulness
        yield CritiqueEvent(faithfulness=score, notes=notes, accepted=accepted)

        rounds = 0
        while not accepted and rounds < self._settings.agent_max_revision_rounds:
            rounds += 1
            revised_q = (
                f"{question}\n\nIMPORTANT: previous draft scored low on faithfulness "
                f"({score:.2f}). Be more conservative; only state claims directly cited."
            )
            draft = await analyze(self._llm, question=revised_q, contexts=ctx_pairs)
            yield DraftEvent(text=draft)
            score, notes = await critique(self._llm, draft=draft, contexts=ctx_pairs)
            accepted = score >= self._settings.agent_critic_min_faithfulness
            yield CritiqueEvent(faithfulness=score, notes=notes, accepted=accepted)

        citations = sorted({acc for acc, _ in ctx_pairs})
        yield FinalAnswerEvent(text=draft, citations=citations)


def _dedupe_contexts(results: list[RetrievalResult]) -> list[tuple[str, str]]:
    seen: set[int] = set()
    out: list[tuple[str, str]] = []
    for r in results:
        if r.chunk_id in seen:
            continue
        seen.add(r.chunk_id)
        out.append((r.accession, r.text))
    return out


async def stream_sse(
    loop: AgentLoop, *, question: str, ticker: str | None = None
) -> AsyncIterator[str]:
    """Wrap AgentLoop output as SSE-formatted strings."""
    async for evt in loop.run(question=question, ticker=ticker):
        yield f"data: {serialize_event(evt)}\n\n"
