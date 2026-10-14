"""Weighted Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

RRF_K = 60


def rrf(
    ranked_lists: Sequence[Sequence[tuple[Hashable, float]]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[tuple[Hashable, float]]:
    """Fuse ranked lists of ``(id, score)``; each list must be sorted best-first.

    fused(id) = sum_i weights[i] / (k + rank_i(id)), rank starting at 1. Only rank
    matters; the original scores are ignored. Duplicate ids within a list count once
    (at their best rank). Ties are broken by first appearance order.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights and ranked_lists must have the same length")

    scores: dict[Hashable, float] = {}
    for lst, w in zip(ranked_lists, weights, strict=True):
        if w == 0:
            continue
        seen: set[Hashable] = set()
        rank = 0
        for item_id, _score in lst:
            if item_id in seen:
                continue
            seen.add(item_id)
            rank += 1
            scores[item_id] = scores.get(item_id, 0.0) + w / (k + rank)

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:limit] if limit is not None else fused
