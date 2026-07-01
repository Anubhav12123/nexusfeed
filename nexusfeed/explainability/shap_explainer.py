"""SHAP explainability — Addition 6.

For each recommendation, compute SHAP values for the top contributing
features, then translate the single highest-magnitude feature into a
human-readable, user-facing sentence ("We recommended this because..."),
while the full attribution vector is exposed to the developer dashboard for
debugging and regulatory-transparency requirements.
"""
from __future__ import annotations

import numpy as np
import shap

from nexusfeed.models.ranking_model import RANKING_FEATURE_NAMES

_FEATURE_EXPLANATIONS = {
    "user_item_dot_product": "you viewed similar content recently",
    "item_freshness_score": "it's new and trending right now",
    "user_item_category_affinity": "you engage a lot with this category",
    "time_decay": "it matches what you usually browse at this time of day",
    "diversity_score": "it broadens your feed with something different",
    "historical_ctr": "other users like it a lot",
    "popularity_score": "it's currently popular across NexusFeed",
}


class ShapExplainer:
    def __init__(self, booster) -> None:  # noqa: ANN001 - lightgbm.Booster
        self.explainer = shap.TreeExplainer(booster)

    def explain_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Returns a (n_samples, n_features) SHAP value matrix."""
        return self.explainer.shap_values(feature_matrix)

    def top_features(self, shap_values: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(shap_values))[:top_k]
        return [(RANKING_FEATURE_NAMES[i], float(shap_values[i])) for i in order]

    def user_facing_explanation(self, shap_values: np.ndarray) -> str:
        top_feature, _ = self.top_features(shap_values, top_k=1)[0]
        return _FEATURE_EXPLANATIONS.get(top_feature, "it matches your personalization profile")

    def developer_attribution(
        self, item_id: str, shap_values: np.ndarray, model_version: str, rank: int
    ) -> dict:
        """Full drill-down record for the developer/admin explainability dashboard."""
        return {
            "item_id": item_id,
            "model_version": model_version,
            "rank": rank,
            "feature_contributions": {
                name: float(value) for name, value in zip(RANKING_FEATURE_NAMES, shap_values, strict=True)
            },
            "top_features": self.top_features(shap_values, top_k=5),
        }

    def detect_false_attribution(
        self,
        shap_values_batch: np.ndarray,
        dominant_feature: str = "popularity_score",
        threshold: float = 0.5,
    ) -> float:
        """Flags the fraction of recommendations where a popularity-style
        feature dominates over genuine personalization signal — this is the
        "31% false attribution rate" finding referenced in the blueprint's
        research-engineer resume bullet.
        """
        if dominant_feature not in RANKING_FEATURE_NAMES:
            raise ValueError(f"unknown feature: {dominant_feature}")
        idx = RANKING_FEATURE_NAMES.index(dominant_feature)
        abs_vals = np.abs(shap_values_batch)
        totals = abs_vals.sum(axis=1) + 1e-9
        dominant_share = abs_vals[:, idx] / totals
        return float(np.mean(dominant_share > threshold))
