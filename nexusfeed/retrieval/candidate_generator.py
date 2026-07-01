"""Top-K candidate retrieval combining ANN search with heuristic rules
(forced exploration for cold-start items, trending injection slots).
"""
from __future__ import annotations

import numpy as np

from nexusfeed.observability.metrics import FAISS_RETRIEVAL_SECONDS
from nexusfeed.retrieval.faiss_index import FaissIndex

TRENDING_INJECTION_POSITIONS = {3, 7}  # Addition 1: trending items at slot 3 and 7


class CandidateGenerator:
    def __init__(self, index: FaissIndex) -> None:
        self.index = index

    async def retrieve(
        self,
        user_embedding: list[float],
        k: int = 1000,
        trending_item_ids: list[str] | None = None,
        forced_exploration_item_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        with FAISS_RETRIEVAL_SECONDS.time():
            query = np.array(user_embedding, dtype=np.float32)
            results = self.index.search(query, k=k)

        seen = {item_id for item_id, _ in results}

        # Addition 3: force a small exploration slice for items that haven't
        # accumulated enough impressions to be reliably scored by ANN.
        for item_id in (forced_exploration_item_ids or [])[:5]:
            if item_id not in seen:
                results.append((item_id, 0.0))
                seen.add(item_id)

        # Addition 1: guarantee trending items appear even if their embedding
        # similarity to this user is weak — popularity is a distinct signal
        # from personal relevance.
        for item_id in (trending_item_ids or [])[:10]:
            if item_id not in seen:
                results.append((item_id, 0.0))
                seen.add(item_id)

        return results
