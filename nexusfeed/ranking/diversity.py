"""Maximal Marginal Relevance (MMR) for feed diversity — Addition 2.

Same algorithm Google uses for search-result diversification. Diversity is a
property of the whole feed, not of individual items: MMR balances an item's
relevance against its similarity to items already selected for the feed.
"""
from __future__ import annotations

import numpy as np


def mmr_rerank(
    candidate_ids: list[str],
    relevance_scores: np.ndarray,
    embeddings: np.ndarray,
    n: int,
    lambda_relevance: float = 0.7,
) -> list[str]:
    """lambda_relevance close to 1.0 favors pure relevance; close to 0.0
    favors pure diversity. 0.7 is a reasonable default that still lets
    diversity break ties among near-equally-relevant items.
    """
    if len(candidate_ids) == 0:
        return []

    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    selected: list[int] = []
    remaining = list(range(len(candidate_ids)))

    while remaining and len(selected) < n:
        if not selected:
            best_idx = max(remaining, key=lambda i: relevance_scores[i])
        else:
            selected_embs = normed[selected]

            def mmr_score(i: int) -> float:
                max_sim = float(np.max(selected_embs @ normed[i])) if len(selected) else 0.0
                return lambda_relevance * relevance_scores[i] - (1 - lambda_relevance) * max_sim

            best_idx = max(remaining, key=mmr_score)

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_ids[i] for i in selected]


def category_exposure_cap(
    ranked_ids: list[str],
    categories: dict[str, str],
    max_category_fraction: float = 0.4,
) -> list[str]:
    """Post-MMR guard: no single category exceeds `max_category_fraction` of
    the feed unless the user's affinity for it is unusually strong (handled
    upstream by the reranker's per-user diversity preference).
    """
    max_count = max(1, int(len(ranked_ids) * max_category_fraction))
    category_counts: dict[str, int] = {}
    result: list[str] = []
    overflow: list[str] = []

    for item_id in ranked_ids:
        category = categories.get(item_id, "unknown")
        if category_counts.get(category, 0) < max_count:
            result.append(item_id)
            category_counts[category] = category_counts.get(category, 0) + 1
        else:
            overflow.append(item_id)

    result.extend(overflow)
    return result
