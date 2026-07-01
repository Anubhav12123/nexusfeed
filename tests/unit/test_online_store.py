import uuid

import fakeredis.aioredis
import pytest

from nexusfeed.exceptions import FeatureNotFoundError
from nexusfeed.features.online_store import OnlineFeatureStore


@pytest.fixture
async def store():
    # decode_responses=True matches the client the app actually constructs
    # in nexusfeed/api/main.py — embeddings must survive this round trip.
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield OnlineFeatureStore(redis)
    await redis.aclose()


@pytest.mark.asyncio
async def test_embedding_roundtrip(store):
    user_id = uuid.uuid4()
    vector = [0.1, 0.2, 0.3] * 42 + [0.4, 0.5]  # 128-dim
    await store.set_user_embedding(user_id, vector)
    result = await store.get_user_embedding(user_id)
    assert len(result) == len(vector)
    assert all(abs(a - b) < 1e-5 for a, b in zip(vector, result, strict=True))


@pytest.mark.asyncio
async def test_missing_embedding_raises(store):
    with pytest.raises(FeatureNotFoundError):
        await store.get_user_embedding(uuid.uuid4())


@pytest.mark.asyncio
async def test_recent_items_capped_at_fifty(store):
    user_id = uuid.uuid4()
    for i in range(60):
        await store.add_recent_item(user_id, uuid.uuid4(), timestamp=float(i))
    recent = await store.get_recent_items(user_id, limit=100)
    assert len(recent) <= 50


@pytest.mark.asyncio
async def test_trending_and_rate_limit(store):
    item_id = uuid.uuid4()
    await store.incr_trending(item_id, by=5.0)
    trending = await store.get_trending()
    assert trending[0][0] == str(item_id)

    allowed, count = await store.rate_limit_hit("key1", window_seconds=60, limit=5)
    assert allowed is True
    assert count == 1
