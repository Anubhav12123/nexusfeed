import numpy as np
import pytest

from nexusfeed.ranking.ranker import Ranker


class _StubModel:
    """Scores by the first feature value — enough to test Ranker's plumbing
    without pulling in a real trained LightGBM booster.
    """

    def score(self, features: np.ndarray) -> np.ndarray:
        return features[:, 0] if len(features) else np.array([])


def test_ranker_builds_feature_matrix_in_declared_order():
    ranker = Ranker(_StubModel())
    features = [{"user_item_dot_product": 0.9, "item_freshness_score": 0.1}]
    matrix = ranker.build_feature_matrix(features)
    assert matrix.shape == (1, 7)
    # matrix is float32; compare with tolerance rather than exact equality
    # since 0.9/0.1 aren't exactly representable and float32 rounds
    # differently than the float64 literals in this test.
    assert matrix[0, 0] == pytest.approx(0.9)
    assert matrix[0, 1] == pytest.approx(0.1)


def test_ranker_scores_and_sorts_descending():
    ranker = Ranker(_StubModel())
    ids = ["a", "b", "c"]
    features = [
        {"user_item_dot_product": 0.1},
        {"user_item_dot_product": 0.9},
        {"user_item_dot_product": 0.5},
    ]
    result = ranker.score(ids, features)
    assert [r.item_id for r in result] == ["b", "c", "a"]


def test_ranker_handles_empty_candidates():
    ranker = Ranker(_StubModel())
    result = ranker.score([], [])
    assert result == []
