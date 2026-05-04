from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from equityiq_agents.events import (
    AgentEvent,
    CritiqueEvent,
    DraftEvent,
    FinalAnswerEvent,
    PlanEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from equityiq_api import app
from equityiq_api.deps import get_agent_loop
from fastapi.testclient import TestClient


def _scripted_events() -> list[AgentEvent]:
    return [
        PlanEvent(subqueries=["q1", "q2"]),
        ToolCallEvent(name="retrieve_filings", args={"query": "q1"}),
        ToolResultEvent(name="retrieve_filings", summary="q1 → 3 chunks", n_results=3),
        ToolCallEvent(name="retrieve_filings", args={"query": "q2"}),
        ToolResultEvent(name="retrieve_filings", summary="q2 → 2 chunks", n_results=2),
        DraftEvent(text="draft text [acc-1]"),
        CritiqueEvent(faithfulness=0.85, notes="ok", accepted=True),
        FinalAnswerEvent(text="final answer [acc-1]", citations=["acc-1"]),
    ]


@pytest.fixture
def fake_loop() -> AsyncMock:
    fake = AsyncMock()

    async def _gen(*, question: str, ticker: str | None = None) -> AsyncIterator[AgentEvent]:
        for e in _scripted_events():
            yield e

    fake.run = _gen
    return fake


@pytest.fixture
def http(fake_loop: AsyncMock):
    app.dependency_overrides[get_agent_loop] = lambda: fake_loop
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_agent_loop, None)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    event: str | None = None
    for line in body.splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event is not None and event != "done":
            data = line.split(":", 1)[1].strip()
            out.append((event, json.loads(data)))
    return out


def test_thesis_stream_emits_full_event_sequence(http: TestClient) -> None:
    resp = http.post("/thesis/stream", json={"ticker": "AAPL", "question": "supply risk?"})
    assert resp.status_code == 200

    parsed = _parse_sse(resp.text)
    types = [e for e, _ in parsed]
    assert types == [
        "plan",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "draft",
        "critique",
        "final",
    ]
    final_event = parsed[-1][1]
    assert final_event["citations"] == ["acc-1"]
    assert "final answer" in final_event["text"]


def test_thesis_stream_validates_ticker_format(http: TestClient) -> None:
    resp = http.post("/thesis/stream", json={"ticker": "lowercase", "question": "x?"})
    assert resp.status_code == 422


def test_thesis_stream_validates_question_min_len(http: TestClient) -> None:
    resp = http.post("/thesis/stream", json={"ticker": "AAPL", "question": "x"})
    assert resp.status_code == 422
