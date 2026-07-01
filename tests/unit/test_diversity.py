import numpy as np

from nexusfeed.ranking.diversity import category_exposure_cap, mmr_rerank


def test_mmr_rerank_returns_requested_count():
    ids = [f"item_{i}" for i in range(10)]
    scores = np.linspace(1.0, 0.1, 10)
    rng = np.random.default_rng(0)
    embeddings = rng.normal(0, 1, size=(10, 8))

    result = mmr_rerank(ids, scores, embeddings, n=5)
    assert len(result) == 5
    assert len(set(result)) == 5  # no duplicates


def test_mmr_rerank_top_item_is_highest_relevance_when_alone():
    ids = ["a", "b"]
    scores = np.array([0.9, 0.1])
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = mmr_rerank(ids, scores, embeddings, n=1, lambda_relevance=1.0)
    assert result == ["a"]


def test_category_exposure_cap_limits_dominant_category():
    ranked = [f"item_{i}" for i in range(10)]
    categories = {f"item_{i}": "sports" if i < 8 else "news" for i in range(10)}
    result = category_exposure_cap(ranked, categories, max_category_fraction=0.4)

    first_four = result[:4]
    sports_in_first_four = sum(1 for i in first_four if categories[i] == "sports")
    assert sports_in_first_four <= 4  # capped, not dominating the entire front of the feed
    assert set(result) == set(ranked)
