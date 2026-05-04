"""Reciprocal Rank Fusion.

RRF score for a doc d:  sum over rankers r of  1 / (k + rank_r(d))

- Robust to score-scale differences (cosine vs BM25 are not comparable raw).
- k=60 is the canonical value from Cormack et al., 2009.
- Missing rankers contribute 0; presence in any ranker still helps.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[int]],
    *,
    k: int = 60,
) -> dict[int, float]:
    """Return doc_id → fused score, sorted descending by score.

    `rankings` is a list of ranked lists of doc_ids (best first).
    """
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
