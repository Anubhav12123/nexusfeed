"""Post-processing: diversity (MMR), freshness boost, seen-item penalty.

Applied after the LightGBM ranker, on the (much smaller) top-N list — this
is the ~1ms step in the feed endpoint latency budget.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from nexusfeed.ranking.diversity import category_exposure_cap, mmr_rerank
from nexusfeed.types import ScoredItem

SEEN_ITEM_PENALTY = 0.5  # items viewed in last 7 days discounted 50%
FRESHNESS_BOOST = 1.10  # new items get a 10% score uplift for their first 24h
NEW_ITEM_WINDOW_HOURS = 24


class Reranker:
    def apply(
        self,
        scored_items: list[ScoredItem],
        n: int,
        seen_items: set[str],
        item_created_at: dict[str, datetime] | None = None,
        item_categories: dict[str, str] | None = None,
        item_embeddings: dict[str, list[float]] | None = None,
        user_diversity_preference: float = 0.7,
        now: datetime | None = None,
    ) -> list[ScoredItem]:
        now = now or datetime.now(timezone.utc)
        item_created_at = item_created_at or {}
        item_categories = item_categories or {}
        item_embeddings = item_embeddings or {}

        adjusted: list[ScoredItem] = []
        for item in scored_items:
            score = item.score
            freshness_boosted = False

            if item.item_id in seen_items:
                score *= SEEN_ITEM_PENALTY

            created_at = item_created_at.get(item.item_id)
            if created_at is not None:
                age_hours = (now - created_at).total_seconds() / 3600.0
                if age_hours <= NEW_ITEM_WINDOW_HOURS:
                    score *= FRESHNESS_BOOST
                    freshness_boosted = True

            adjusted.append(
                ScoredItem(
                    item_id=item.item_id,
                    score=score,
                    category=item_categories.get(item.item_id, item.category),
                    is_trending=item.is_trending,
                    freshness_boosted=freshness_boosted,
                )
            )

        adjusted.sort(key=lambda i: i.score, reverse=True)

        if item_embeddings:
            ids_with_embeddings = [i.item_id for i in adjusted if i.item_id in item_embeddings]
            if ids_with_embeddings:
                embeddings = np.array([item_embeddings[i] for i in ids_with_embeddings])
                relevance = np.array(
                    [next(i.score for i in adjusted if i.item_id == iid) for iid in ids_with_embeddings]
                )
                mmr_order = mmr_rerank(
                    ids_with_embeddings,
                    relevance,
                    embeddings,
                    n=len(ids_with_embeddings),
                    lambda_relevance=user_diversity_preference,
                )
                by_id = {item.item_id: item for item in adjusted}
                remaining = [item for item in adjusted if item.item_id not in item_embeddings]
                adjusted = [by_id[iid] for iid in mmr_order] + remaining

        ordered_ids = category_exposure_cap(
            [i.item_id for i in adjusted], item_categories, max_category_fraction=0.4
        )
        by_id = {item.item_id: item for item in adjusted}
        final = [by_id[iid] for iid in ordered_ids][:n]
        return final
