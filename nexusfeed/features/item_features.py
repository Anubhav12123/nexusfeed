"""Item embedding, freshness score, and popularity counter computation."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import UUID

from nexusfeed.features.online_store import OnlineFeatureStore

FRESHNESS_HALF_LIFE_HOURS = 48.0
NEW_ITEM_IMPRESSION_THRESHOLD = 100
FORCED_EXPLORATION_RATE = 0.05  # Addition 3: new items get a 5% forced-exploration slot


def compute_freshness_score(created_at: datetime, now: datetime | None = None) -> float:
    """Exponential decay: score halves every FRESHNESS_HALF_LIFE_HOURS."""
    now = now or datetime.now(timezone.utc)
    age_hours = max((now - created_at).total_seconds() / 3600.0, 0.0)
    decay = math.pow(0.5, age_hours / FRESHNESS_HALF_LIFE_HOURS)
    return decay


def compute_composite_score(freshness: float, popularity: float, freshness_weight: float = 0.3) -> float:
    return freshness_weight * freshness + (1 - freshness_weight) * popularity


class ItemFeatureService:
    def __init__(self, online_store: OnlineFeatureStore) -> None:
        self.online_store = online_store

    async def refresh_item_score(self, item_id: UUID, created_at: datetime, popularity: float) -> float:
        freshness = compute_freshness_score(created_at)
        composite = compute_composite_score(freshness, popularity)
        await self.online_store.set_item_score(item_id, composite)
        return composite

    async def is_forced_exploration_candidate(self, item_id: UUID, impression_count: int) -> bool:
        """Addition 3 cold-start-for-items: brand-new items get a random
        exploration slot regardless of predicted relevance, so the model
        eventually gets signal on them instead of being starved by low priors.
        """
        if impression_count >= NEW_ITEM_IMPRESSION_THRESHOLD:
            return False
        import random

        return random.random() < FORCED_EXPLORATION_RATE

    async def get_trending_flag(self, item_id: UUID, z_score_threshold: float = 3.0) -> bool:
        """Addition 1: item is 'trending' if its 1h interaction rate is more
        than `z_score_threshold` standard deviations above the rolling mean
        across all items currently in the trending zset.
        """
        trending = await self.online_store.get_trending(limit=1000)
        if len(trending) < 5:
            return False
        scores = [score for _, score in trending]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        item_score = next((s for iid, s in trending if iid == str(item_id)), None)
        if item_score is None:
            return False
        return (item_score - mean) / std > z_score_threshold
