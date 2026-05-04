from __future__ import annotations

import json
import re
from dataclasses import dataclass

from equityiq_llm import OllamaClient
from equityiq_llm.config import ModelTier

from equityiq_eval.types import JudgeScore

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')
_RATIONALE_RE = re.compile(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _parse_judge_json(raw: str) -> tuple[float, str]:
    """Robust score parser. Tries JSON first, then regex fallback."""
    try:
        obj = json.loads(raw)
        score = float(obj.get("score", 0.0))
        rationale = str(obj.get("rationale", ""))
        return _clamp(score), rationale
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = _SCORE_RE.search(raw)
    if not m:
        return 0.0, raw[:200]
    score = _clamp(float(m.group(1)))
    rm = _RATIONALE_RE.search(raw)
    rationale = rm.group(1) if rm else raw[:200]
    return score, rationale


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


_FAITHFULNESS_PROMPT = """You judge whether an answer is faithful to provided source contexts.
Score 1.0 if every factual claim in the answer is directly supported by the contexts.
Score 0.0 if the answer contains hallucinations not in the contexts.

Question: {question}

Contexts:
{contexts}

Answer:
{answer}

Respond with strict JSON: {{"score": <float 0..1>, "rationale": "<one sentence>"}}"""


_RELEVANCE_PROMPT = """You judge whether an answer addresses the question.
Score 1.0 if the answer fully addresses the question. Score 0.0 if it is off-topic.

Question: {question}

Answer:
{answer}

Reference answer (for guidance, not strict matching):
{reference}

Respond with strict JSON: {{"score": <float 0..1>, "rationale": "<one sentence>"}}"""


@dataclass(frozen=True, slots=True)
class _JudgePrompt:
    metric: str
    template: str


class LLMJudge:
    """LLM-as-judge wrapper. Uses ModelTier.JUDGE."""

    def __init__(self, llm: OllamaClient, *, temperature: float = 0.0) -> None:
        self._llm = llm
        self._temperature = temperature

    async def score(
        self,
        prompt: _JudgePrompt,
        *,
        question: str,
        answer: str,
        contexts: str = "",
        reference: str = "",
    ) -> JudgeScore:
        body = prompt.template.format(
            question=question,
            answer=answer,
            contexts=contexts,
            reference=reference,
        )
        raw = await self._llm.generate(
            prompt=body,
            tier=ModelTier.JUDGE,
            options={"temperature": self._temperature},
            format="json",
        )
        score, rationale = _parse_judge_json(raw)
        return JudgeScore(metric=prompt.metric, score=score, rationale=rationale)


_FAITHFULNESS = _JudgePrompt(metric="faithfulness", template=_FAITHFULNESS_PROMPT)
_RELEVANCE = _JudgePrompt(metric="answer_relevance", template=_RELEVANCE_PROMPT)


async def faithfulness(
    judge: LLMJudge, *, question: str, answer: str, contexts: list[str]
) -> JudgeScore:
    joined = "\n---\n".join(contexts) if contexts else "(no contexts)"
    return await judge.score(_FAITHFULNESS, question=question, answer=answer, contexts=joined)


async def answer_relevance(
    judge: LLMJudge, *, question: str, answer: str, reference: str
) -> JudgeScore:
    return await judge.score(_RELEVANCE, question=question, answer=answer, reference=reference)


def context_precision(
    *, retrieved_accessions: list[str], expected_accessions: list[str]
) -> JudgeScore:
    """Deterministic precision: fraction of retrieved accessions in expected set.

    No LLM call — pure set math. Returns 1.0 if expected set empty (nothing to verify).
    """
    if not retrieved_accessions:
        return JudgeScore(metric="context_precision", score=0.0, rationale="no retrieved chunks")
    if not expected_accessions:
        return JudgeScore(
            metric="context_precision", score=1.0, rationale="no expected accessions specified"
        )
    expected = set(expected_accessions)
    hits = sum(1 for a in retrieved_accessions if a in expected)
    score = hits / len(retrieved_accessions)
    return JudgeScore(
        metric="context_precision",
        score=score,
        rationale=f"{hits}/{len(retrieved_accessions)} retrieved in expected set",
    )
