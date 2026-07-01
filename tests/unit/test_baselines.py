import numpy as np

from nexusfeed.baselines.collaborative_filtering import CollaborativeFilteringBaseline
from nexusfeed.baselines.popularity_baseline import PopularityBaseline
from nexusfeed.baselines.random_baseline import RandomBaseline


def test_random_baseline_returns_requested_count_from_catalog():
    items = [f"item_{i}" for i in range(20)]
    baseline = RandomBaseline(items)
    result = baseline.recommend("user_1", n=5)
    assert len(result) == 5
    assert set(result).issubset(set(items))


def test_random_baseline_caps_at_catalog_size():
    items = [f"item_{i}" for i in range(3)]
    baseline = RandomBaseline(items)
    result = baseline.recommend("user_1", n=10)
    assert len(result) == 3


def test_popularity_baseline_ranks_by_count_descending():
    counts = {"a": 5, "b": 50, "c": 1}
    baseline = PopularityBaseline(counts)
    result = baseline.recommend("user_1", n=3)
    assert result == ["b", "a", "c"]


def test_collaborative_filtering_fit_and_recommend():
    rng = np.random.default_rng(0)
    num_users, num_items = 10, 15
    matrix = rng.integers(0, 2, size=(num_users, num_items)).astype(np.float32)
    user_ids = [f"u{i}" for i in range(num_users)]
    item_ids = [f"i{i}" for i in range(num_items)]

    cf = CollaborativeFilteringBaseline(num_factors=4).fit(matrix, user_ids, item_ids)
    result = cf.recommend("u0", n=5)
    assert len(result) == 5
    assert set(result).issubset(set(item_ids))


def test_collaborative_filtering_excludes_seen_items():
    rng = np.random.default_rng(1)
    matrix = rng.integers(0, 2, size=(5, 8)).astype(np.float32)
    user_ids = [f"u{i}" for i in range(5)]
    item_ids = [f"i{i}" for i in range(8)]

    cf = CollaborativeFilteringBaseline(num_factors=3).fit(matrix, user_ids, item_ids)
    result = cf.recommend("u0", n=8, exclude={"i0", "i1"})
    assert "i0" not in result
    assert "i1" not in result


def test_collaborative_filtering_unknown_user_returns_empty():
    matrix = np.zeros((2, 2), dtype=np.float32)
    cf = CollaborativeFilteringBaseline(num_factors=2).fit(matrix, ["u0", "u1"], ["i0", "i1"])
    assert cf.recommend("does_not_exist", n=5) == []
