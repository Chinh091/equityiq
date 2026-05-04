from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from equityiq_agents.roles import _parse_critique, _parse_subqueries, analyze, critique, plan


def test_parse_subqueries_strict_json() -> None:
    out = _parse_subqueries('{"subqueries": ["a", "b", "c"]}')
    assert out == ["a", "b", "c"]


def test_parse_subqueries_drops_blanks_and_non_strings() -> None:
    out = _parse_subqueries('{"subqueries": ["a", "", null, "c"]}')
    assert out == ["a", "c"]


def test_parse_subqueries_garbage_returns_empty() -> None:
    assert _parse_subqueries("not json") == []
    assert _parse_subqueries('{"other": []}') == []


def test_parse_critique_strict() -> None:
    score, notes = _parse_critique('{"faithfulness": 0.7, "notes": "ok"}')
    assert score == 0.7
    assert notes == "ok"


def test_parse_critique_clamps() -> None:
    score, _ = _parse_critique('{"faithfulness": 2.0, "notes": "x"}')
    assert score == 1.0
    score, _ = _parse_critique('{"faithfulness": -1, "notes": "x"}')
    assert score == 0.0


def test_parse_critique_falls_back_to_regex() -> None:
    score, _ = _parse_critique('garbage prefix "faithfulness": 0.42 trailing')
    assert score == 0.42


@pytest.mark.asyncio
async def test_plan_falls_back_to_original_question_on_empty() -> None:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"subqueries": []}')
    out = await plan(llm, question="what?", max_subqueries=4)
    assert out == ["what?"]


@pytest.mark.asyncio
async def test_plan_truncates_to_max() -> None:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"subqueries": ["a","b","c","d","e","f"]}')
    out = await plan(llm, question="q", max_subqueries=3)
    assert out == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_analyze_inlines_contexts_with_accessions() -> None:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="answer text")
    out = await analyze(
        llm, question="q", contexts=[("acc-A", "ctx A"), ("acc-B", "ctx B")]
    )
    assert out == "answer text"
    prompt = llm.generate.await_args.kwargs["prompt"]
    assert "[acc-A]" in prompt
    assert "ctx A" in prompt
    assert "[acc-B]" in prompt


@pytest.mark.asyncio
async def test_critique_uses_judge_tier() -> None:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"faithfulness": 0.6, "notes": "n"}')
    score, notes = await critique(llm, draft="d", contexts=[("acc", "c")])
    assert score == 0.6
    assert notes == "n"
    assert llm.generate.await_args.kwargs["tier"].value == "judge"
