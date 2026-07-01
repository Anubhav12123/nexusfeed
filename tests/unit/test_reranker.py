from datetime import datetime, timedelta, timezone

from nexusfeed.ranking.reranker import Reranker
from nexusfeed.types import ScoredItem


def test_seen_item_penalty_applied():
    items = [ScoredItem(item_id="a", score=1.0), ScoredItem(item_id="b", score=0.9)]
    reranker = Reranker()
    result = reranker.apply(items, n=2, seen_items={"a"})

    result_by_id = {i.item_id: i.score for i in result}
    assert result_by_id["a"] < 1.0  # penalized
    assert result_by_id["b"] == 0.9  # untouched


def test_freshness_boost_applied_to_new_items():
    now = datetime.now(timezone.utc)
    items = [ScoredItem(item_id="new", score=0.5), ScoredItem(item_id="old", score=0.5)]
    created_at = {"new": now - timedelta(hours=1), "old": now - timedelta(hours=100)}

    reranker = Reranker()
    result = reranker.apply(items, n=2, seen_items=set(), item_created_at=created_at, now=now)

    result_by_id = {i.item_id: i.score for i in result}
    assert result_by_id["new"] > result_by_id["old"]


def test_apply_respects_n_limit():
    items = [ScoredItem(item_id=f"item_{i}", score=1.0 - i * 0.01) for i in range(50)]
    reranker = Reranker()
    result = reranker.apply(items, n=10, seen_items=set())
    assert len(result) == 10
