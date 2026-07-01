import uuid

from nexusfeed.experiments.bucketing import assign_variant, bucket_for, is_in_holdback


def test_bucket_is_deterministic():
    user_id = uuid.uuid4()
    b1 = bucket_for(user_id, "exp_a")
    b2 = bucket_for(user_id, "exp_a")
    assert b1 == b2
    assert 0 <= b1 < 100


def test_bucket_differs_across_experiments_usually():
    user_id = uuid.uuid4()
    buckets = {bucket_for(user_id, f"exp_{i}") for i in range(20)}
    assert len(buckets) > 1  # extremely unlikely all 20 collide


def test_assign_variant_ranges():
    assert assign_variant(10, (0, 50), (50, 100)) == "control"
    assert assign_variant(75, (0, 50), (50, 100)) == "treatment"
    assert assign_variant(99, (0, 50), (50, 90)) == "unassigned"


def test_holdback_is_stable_and_small_fraction():
    users = [uuid.uuid4() for _ in range(5000)]
    holdback_count = sum(1 for u in users if is_in_holdback(u, holdback_fraction=0.05))
    fraction = holdback_count / len(users)
    assert 0.03 < fraction < 0.07  # should be close to 5%

    # stability: calling twice for the same user gives the same result
    assert is_in_holdback(users[0]) == is_in_holdback(users[0])
