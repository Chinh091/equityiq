from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from equityiq_eval.judge import (
    LLMJudge,
    _parse_judge_json,
    answer_relevance,
    context_precision,
    faithfulness,
)


def test_parse_judge_json_strict() -> None:
    score, rationale = _parse_judge_json('{"score": 0.85, "rationale": "supported"}')
    assert score == 0.85
    assert rationale == "supported"


def test_parse_judge_json_clamps_out_of_range() -> None:
    score, _ = _parse_judge_json('{"score": 1.5, "rationale": "x"}')
    assert score == 1.0
    score, _ = _parse_judge_json('{"score": -0.2, "rationale": "x"}')
    assert score == 0.0


def test_parse_judge_json_falls_back_to_regex() -> None:
    raw = 'Here is my judgment: {"score": 0.4, "rationale": "weak"} ok?'
    score, rationale = _parse_judge_json(raw)
    assert score == 0.4
    assert "weak" in rationale


def test_parse_judge_json_handles_garbage() -> None:
    score, rationale = _parse_judge_json("totally not json")
    assert score == 0.0
    assert rationale  # non-empty


def test_context_precision_no_retrieved_returns_zero() -> None:
    s = context_precision(retrieved_accessions=[], expected_accessions=["a"])
    assert s.score == 0.0
    assert s.metric == "context_precision"


def test_context_precision_no_expected_returns_one() -> None:
    s = context_precision(retrieved_accessions=["a", "b"], expected_accessions=[])
    assert s.score == 1.0


def test_context_precision_partial_hit() -> None:
    s = context_precision(
        retrieved_accessions=["a", "b", "c", "d"],
        expected_accessions=["a", "c"],
    )
    assert s.score == 0.5


def test_context_precision_all_hit() -> None:
    s = context_precision(
        retrieved_accessions=["a", "b"],
        expected_accessions=["a", "b", "c"],
    )
    assert s.score == 1.0


@pytest.mark.asyncio
async def test_faithfulness_invokes_judge_tier() -> None:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"score": 0.9, "rationale": "supported"}')
    judge = LLMJudge(llm)
    score = await faithfulness(judge, question="q", answer="a", contexts=["ctx1", "ctx2"])
    assert score.metric == "faithfulness"
    assert score.score == 0.9
    call = llm.generate.await_args
    assert call.kwargs["tier"].value == "judge"
    assert "ctx1" in call.kwargs["prompt"]
    assert "ctx2" in call.kwargs["prompt"]


@pytest.mark.asyncio
async def test_answer_relevance_includes_reference() -> None:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"score": 0.7, "rationale": "ok"}')
    judge = LLMJudge(llm)
    score = await answer_relevance(judge, question="q", answer="a", reference="ref-text")
    assert score.metric == "answer_relevance"
    assert score.score == 0.7
    assert "ref-text" in llm.generate.await_args.kwargs["prompt"]


@pytest.mark.asyncio
async def test_faithfulness_handles_empty_contexts() -> None:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"score": 0.0, "rationale": "no ctx"}')
    judge = LLMJudge(llm)
    score = await faithfulness(judge, question="q", answer="a", contexts=[])
    assert score.score == 0.0
    assert "no contexts" in llm.generate.await_args.kwargs["prompt"]
