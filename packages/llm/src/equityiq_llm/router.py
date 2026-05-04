from __future__ import annotations

from dataclasses import dataclass

from equityiq_llm.config import ModelTier


@dataclass(frozen=True, slots=True)
class RouteDecision:
    tier: ModelTier
    reason: str


class ModelRouter:
    """Cheap first-pass router: picks model tier from query signals.

    Phase 1: rule-based (length, keyword complexity, requires-citations).
    Phase 5: replace with learned router or quality-aware A/B.
    """

    HARD_MARKERS: tuple[str, ...] = (
        "compare",
        "contradiction",
        "thesis",
        "explain why",
        "step by step",
        "valuation",
    )

    def __init__(
        self,
        *,
        long_query_chars: int = 600,
        force_primary: bool = False,
    ) -> None:
        self._long_query_chars = long_query_chars
        self._force_primary = force_primary

    def route(self, query: str, *, requires_citations: bool = False) -> RouteDecision:
        if self._force_primary:
            return RouteDecision(ModelTier.PRIMARY, "force_primary flag set")
        q = query.lower()
        if requires_citations:
            return RouteDecision(ModelTier.PRIMARY, "citations required")
        if len(query) >= self._long_query_chars:
            return RouteDecision(ModelTier.PRIMARY, "long query")
        if any(m in q for m in self.HARD_MARKERS):
            return RouteDecision(ModelTier.PRIMARY, "complexity marker matched")
        return RouteDecision(ModelTier.FALLBACK, "default fallback tier")
