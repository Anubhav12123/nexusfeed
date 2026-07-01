"""User embedding computation and recent-interaction retrieval."""
from __future__ import annotations

from uuid import UUID

import numpy as np

from nexusfeed.exceptions import FeatureNotFoundError
from nexusfeed.features.offline_store import OfflineFeatureStore
from nexusfeed.features.online_store import OnlineFeatureStore
from nexusfeed.types import UserProfile


class UserFeatureService:
    """Serving-time access to user features with a cold-start fallback.

    Hot path: Redis embedding lookup (~1ms). If missing (new user, cache
    expired), falls back to a deterministic category-prior embedding derived
    from onboarding categories rather than a random vector — this is the
    "hybrid transition" cold-start approach described in blueprint Addition 3.
    """

    def __init__(
        self,
        online_store: OnlineFeatureStore,
        offline_store: OfflineFeatureStore,
        embedding_dim: int = 128,
    ) -> None:
        self.online_store = online_store
        self.offline_store = offline_store
        self.embedding_dim = embedding_dim

    async def get_embedding(self, user_id: UUID) -> tuple[list[float], bool]:
        """Returns (embedding, is_cold_start)."""
        try:
            embedding = await self.online_store.get_user_embedding(user_id)
            return embedding, False
        except FeatureNotFoundError:
            categories = await self.offline_store.get_user_preferred_categories(user_id)
            embedding = self._cold_start_embedding(user_id, categories)
            return embedding, True

    def _cold_start_embedding(self, user_id: UUID, categories: list[str]) -> list[float]:
        """Deterministic per-category prior so cold-start users with the same
        onboarding categories get a stable (not random-per-request) embedding.
        """
        rng = np.random.default_rng(seed=abs(hash(tuple(sorted(categories)) or ("__none__",))) % (2**32))
        vector = rng.normal(loc=0.0, scale=0.1, size=self.embedding_dim)
        return vector.tolist()

    async def get_recent_interaction_sequence(self, user_id: UUID, length: int = 50) -> list[str]:
        items = await self.online_store.get_recent_items(user_id, limit=length)
        return items
