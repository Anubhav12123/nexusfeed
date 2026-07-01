"""Baseline 3: collaborative filtering via matrix factorization (ALS-style).

Implemented with plain NumPy SVD over the user-item interaction matrix —
enough to serve as a fair, classic baseline for the two-tower model without
pulling in a heavyweight CF library.
"""
from __future__ import annotations

import numpy as np


class CollaborativeFilteringBaseline:
    name = "collaborative_filtering"

    def __init__(self, num_factors: int = 32) -> None:
        self.num_factors = num_factors
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_index: dict[str, int] = {}
        self.item_index: dict[str, int] = {}

    def fit(self, interaction_matrix: np.ndarray, user_ids: list[str], item_ids: list[str]) -> "CollaborativeFilteringBaseline":
        self.user_index = {uid: i for i, uid in enumerate(user_ids)}
        self.item_index = {iid: i for i, iid in enumerate(item_ids)}

        u, s, vt = np.linalg.svd(interaction_matrix, full_matrices=False)
        k = min(self.num_factors, len(s))
        self.user_factors = u[:, :k] * s[:k]
        self.item_factors = vt[:k, :].T
        return self

    def recommend(self, user_id: str, n: int = 20, exclude: set[str] | None = None) -> list[str]:
        if self.user_factors is None or user_id not in self.user_index:
            return []
        exclude = exclude or set()
        user_vec = self.user_factors[self.user_index[user_id]]
        scores = self.item_factors @ user_vec
        ranked_indices = np.argsort(-scores)

        id_by_index = {idx: iid for iid, idx in self.item_index.items()}
        results = []
        for idx in ranked_indices:
            item_id = id_by_index[idx]
            if item_id in exclude:
                continue
            results.append(item_id)
            if len(results) >= n:
                break
        return results
