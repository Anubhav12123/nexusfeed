from datetime import datetime, timedelta, timezone

from nexusfeed.features.item_features import compute_composite_score, compute_freshness_score


def test_freshness_score_decays_over_time():
    now = datetime.now(timezone.utc)
    fresh = compute_freshness_score(now, now=now)
    old = compute_freshness_score(now - timedelta(hours=200), now=now)
    assert fresh > old
    assert 0.0 <= old <= 1.0


def test_freshness_score_half_life():
    now = datetime.now(timezone.utc)
    half_life_ago = now - timedelta(hours=48)
    score = compute_freshness_score(half_life_ago, now=now)
    assert abs(score - 0.5) < 0.01


def test_composite_score_weights_freshness_and_popularity():
    score = compute_composite_score(freshness=1.0, popularity=0.0, freshness_weight=0.3)
    assert abs(score - 0.3) < 1e-9

    score2 = compute_composite_score(freshness=0.0, popularity=1.0, freshness_weight=0.3)
    assert abs(score2 - 0.7) < 1e-9
