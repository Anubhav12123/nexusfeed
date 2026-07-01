"""Scores ANN candidates using the LightGBM ONNX re-ranking model."""
from __future__ import annotations

import numpy as np

from nexusfeed.models.ranking_model import RANKING_FEATURE_NAMES
from nexusfeed.observability.metrics import PREDICTION_SCORE, RANKING_SECONDS
from nexusfeed.types import ScoredItem


class Ranker:
    def __init__(self, model) -> None:  # noqa: ANN001 - RankingModel or OnnxRankingModel, both expose .score
        self.model = model

    def build_feature_matrix(self, candidate_features: list[dict[str, float]]) -> np.ndarray:
        return np.array(
            [[feat.get(name, 0.0) for name in RANKING_FEATURE_NAMES] for feat in candidate_features],
            dtype=np.float32,
        )

    def score(self, candidate_ids: list[str], candidate_features: list[dict[str, float]]) -> list[ScoredItem]:
        with RANKING_SECONDS.time():
            matrix = self.build_feature_matrix(candidate_features)
            scores = self.model.score(matrix) if len(matrix) else np.array([])

        scored_items = []
        for item_id, score in zip(candidate_ids, scores, strict=True):
            PREDICTION_SCORE.observe(float(score))
            scored_items.append(ScoredItem(item_id=item_id, score=float(score)))
        return sorted(scored_items, key=lambda i: i.score, reverse=True)
