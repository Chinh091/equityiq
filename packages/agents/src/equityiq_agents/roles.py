from __future__ import annotations

import json
import re
from typing import Any

from equityiq_llm import LLMClient
from equityiq_llm.config import ModelTier

PLANNER_SYSTEM = """You decompose an equity research question into 1-{max_subqueries} focused
sub-queries that can each be answered from SEC filings (10-K, 10-Q, 8-K).

Respond with strict JSON: {{"subqueries": ["...", "..."]}}.
Be concrete: name the metric, segment, or risk factor. Avoid yes/no questions."""


ANALYST_SYSTEM = """You write a concise, well-structured analyst note answering the user's
question, using ONLY the provided context excerpts. Each factual claim must be
followed by a citation in the form [accession]. If the contexts don't support a
claim, say "the filings do not address this" rather than inventing facts."""


CRITIC_SYSTEM = """You judge whether the analyst draft is faithful to the provided contexts.
Respond with strict JSON: {"faithfulness": <0..1>, "notes": "<one sentence>"}.
Faithfulness 1.0 = every claim cited and supported. Faithfulness 0.0 = hallucinations."""


_SCORE_RE = re.compile(r'"faithfulness"\s*:\s*([0-9]*\.?[0-9]+)')


async def plan(llm: LLMClient, *, question: str, max_subqueries: int) -> list[str]:
    raw = await llm.generate(
        prompt=question,
        tier=ModelTier.PRIMARY,
        system=PLANNER_SYSTEM.format(max_subqueries=max_subqueries),
        format="json",
        options={"temperature": 0.2},
    )
    subs = _parse_subqueries(raw)
    return subs[:max_subqueries] if subs else [question]


def _parse_subqueries(raw: str) -> list[str]:
    try:
        obj: Any = json.loads(raw)
        subs = obj.get("subqueries")
        if isinstance(subs, list):
            return [str(s) for s in subs if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return []


async def analyze(llm: LLMClient, *, question: str, contexts: list[tuple[str, str]]) -> str:
    """Contexts: list of (accession, text)."""
    formatted = "\n\n".join(f"[{acc}]\n{txt}" for acc, txt in contexts)
    prompt = f"Question: {question}\n\nContexts:\n{formatted}\n\nWrite the analyst note now."
    return await llm.generate(
        prompt=prompt,
        tier=ModelTier.PRIMARY,
        system=ANALYST_SYSTEM,
        options={"temperature": 0.3},
    )


async def critique(
    llm: LLMClient, *, draft: str, contexts: list[tuple[str, str]]
) -> tuple[float, str]:
    formatted = "\n\n".join(f"[{acc}]\n{txt}" for acc, txt in contexts)
    prompt = f"Draft:\n{draft}\n\nContexts:\n{formatted}"
    raw = await llm.generate(
        prompt=prompt,
        tier=ModelTier.JUDGE,
        system=CRITIC_SYSTEM,
        format="json",
        options={"temperature": 0.0},
    )
    return _parse_critique(raw)


def _parse_critique(raw: str) -> tuple[float, str]:
    try:
        obj = json.loads(raw)
        score = max(0.0, min(1.0, float(obj.get("faithfulness", 0.0))))
        notes = str(obj.get("notes", ""))
        return score, notes
    except (json.JSONDecodeError, ValueError, TypeError):
        m = _SCORE_RE.search(raw)
        score = max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
        return score, raw[:200]
