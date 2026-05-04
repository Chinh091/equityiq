from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from equityiq_agents import (
    AgentLoop,
    AgentSettings,
    CritiqueEvent,
    DraftEvent,
    FinalAnswerEvent,
    PlanEvent,
    RetrieveTool,
    ToolCallEvent,
    ToolResultEvent,
)
from equityiq_agents.loop import stream_sse
from equityiq_retrieval.types import RetrievalResult


def _retrieval(chunk_id: int, accession: str = "acc-1", text: str = "ctx") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        filing_id=1,
        section_id=1,
        item_code="1A",
        text=text,
        ticker="AAPL",
        accession=accession,
        form_type="10-K",
        filed_at=datetime(2024, 1, 1),
        source_url="https://example.test/f.htm",
        chunk_ord=0,
        rerank_score=0.9,
    )


class _FakeLLM:
    """Predictable LLM that cycles through scripted responses by tier."""

    def __init__(self, responses: dict[str, list[str]]) -> None:
        self._responses = {k: list(v) for k, v in responses.items()}

    async def generate(
        self, *, prompt: str, tier, system=None, options=None, format=None
    ) -> str:
        bucket = self._responses.get(tier.value)
        if not bucket:
            raise RuntimeError(f"no fake response for tier {tier}")
        return bucket.pop(0)


def _retriever_returning(*results_per_call: list[RetrievalResult]) -> AsyncMock:
    retriever = AsyncMock()
    retriever.search = AsyncMock(side_effect=list(results_per_call))
    return retriever


@pytest.mark.asyncio
async def test_loop_emits_full_event_sequence() -> None:
    llm = _FakeLLM(
        {
            "primary": [
                '{"subqueries": ["q1", "q2"]}',
                "Apple discloses supply concentration [acc-1].",
            ],
            "judge": ['{"faithfulness": 0.9, "notes": "ok"}'],
        }
    )
    retriever = _retriever_returning(
        [_retrieval(1, "acc-1")],
        [_retrieval(2, "acc-1")],
    )
    tool = RetrieveTool(retriever)
    loop = AgentLoop(llm=llm, retrieve=tool)

    events = [e async for e in loop.run(question="What is Apple's supply risk?")]

    types = [type(e) for e in events]
    assert types == [
        PlanEvent,
        ToolCallEvent,
        ToolResultEvent,
        ToolCallEvent,
        ToolResultEvent,
        DraftEvent,
        CritiqueEvent,
        FinalAnswerEvent,
    ]
    plan_event = events[0]
    assert isinstance(plan_event, PlanEvent)
    assert plan_event.subqueries == ["q1", "q2"]

    final = events[-1]
    assert isinstance(final, FinalAnswerEvent)
    assert final.citations == ["acc-1"]
    assert "supply" in final.text


@pytest.mark.asyncio
async def test_loop_revises_on_low_faithfulness() -> None:
    llm = _FakeLLM(
        {
            "primary": [
                '{"subqueries": ["q1"]}',
                "draft 1 - hallucinated stuff",
                "draft 2 - more conservative [acc-1]",
            ],
            "judge": [
                '{"faithfulness": 0.2, "notes": "hallucinations"}',
                '{"faithfulness": 0.8, "notes": "good"}',
            ],
        }
    )
    retriever = _retriever_returning([_retrieval(1, "acc-1")])
    loop = AgentLoop(
        llm=llm,
        retrieve=RetrieveTool(retriever),
        settings=AgentSettings(agent_max_revision_rounds=1, agent_critic_min_faithfulness=0.6),
    )

    events = [e async for e in loop.run(question="q?")]
    drafts = [e for e in events if isinstance(e, DraftEvent)]
    critiques = [e for e in events if isinstance(e, CritiqueEvent)]
    assert len(drafts) == 2
    assert len(critiques) == 2
    assert critiques[0].accepted is False
    assert critiques[1].accepted is True

    final = events[-1]
    assert isinstance(final, FinalAnswerEvent)
    assert "conservative" in final.text


@pytest.mark.asyncio
async def test_loop_dedupes_contexts_by_chunk_id() -> None:
    llm = _FakeLLM(
        {
            "primary": [
                '{"subqueries": ["q1", "q2"]}',
                "ans [acc-1] [acc-2]",
            ],
            "judge": ['{"faithfulness": 0.9, "notes": "ok"}'],
        }
    )
    # Same chunk_id=1 returned by both sub-queries; chunk_id=2 only from second.
    retriever = _retriever_returning(
        [_retrieval(1, "acc-1")],
        [_retrieval(1, "acc-1"), _retrieval(2, "acc-2")],
    )
    loop = AgentLoop(llm=llm, retrieve=RetrieveTool(retriever))

    events = [e async for e in loop.run(question="q?")]
    final = events[-1]
    assert isinstance(final, FinalAnswerEvent)
    assert final.citations == ["acc-1", "acc-2"]


@pytest.mark.asyncio
async def test_stream_sse_wraps_events_as_data_lines() -> None:
    llm = _FakeLLM(
        {
            "primary": ['{"subqueries": ["q1"]}', "answer [acc-1]"],
            "judge": ['{"faithfulness": 0.9, "notes": "ok"}'],
        }
    )
    retriever = _retriever_returning([_retrieval(1, "acc-1")])
    loop = AgentLoop(llm=llm, retrieve=RetrieveTool(retriever))

    chunks = [c async for c in stream_sse(loop, question="q?")]
    assert all(c.startswith("data: ") for c in chunks)
    assert all(c.endswith("\n\n") for c in chunks)
    # Final chunk should encode the FinalAnswerEvent.
    assert '"type": "final"' in chunks[-1]
