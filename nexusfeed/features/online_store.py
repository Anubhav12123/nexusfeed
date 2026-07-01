"""Redis-backed real-time feature read/write operations.

Key patterns (see blueprint section 4 "Redis Data Structures"):
  user:{id}:embedding    hash   128-dim embedding vector          TTL 1h
  user:{id}:recent       zset   item_ids scored by timestamp      TTL 24h
  user:{id}:experiment   string experiment bucket assignment      TTL 7d
  item:{id}:score        string freshness + popularity composite  TTL 5m
  trending:global        zset   item_ids scored by 1h interaction TTL 15m
  ratelimit:{key}:{win}  string request count in sliding window   TTL 1m
"""
from __future__ import annotations

import base64
import struct
import time
from uuid import UUID

from redis.asyncio import Redis

from nexusfeed.config import Settings, get_settings
from nexusfeed.exceptions import FeatureNotFoundError

EMBEDDING_DTYPE = "f"  # 4-byte float, matches struct format code


def _pack_embedding(vector: list[float]) -> str:
    """Base64-encode the packed floats so the value survives a UTF-8 round
    trip regardless of whether the Redis client was constructed with
    decode_responses=True (the app default) or False — raw struct bytes
    would otherwise be corrupted by decode_responses=True trying to decode
    arbitrary binary data as UTF-8.
    """
    raw = struct.pack(f"{len(vector)}{EMBEDDING_DTYPE}", *vector)
    return base64.b64encode(raw).decode("ascii")


def _unpack_embedding(raw: str | bytes) -> list[float]:
    decoded = base64.b64decode(raw)
    n = len(decoded) // 4
    return list(struct.unpack(f"{n}{EMBEDDING_DTYPE}", decoded))


class OnlineFeatureStore:
    def __init__(self, redis: Redis, settings: Settings | None = None) -> None:
        self.redis = redis
        self.settings = settings or get_settings()

    # ---- user embeddings ----------------------------------------------
    async def set_user_embedding(self, user_id: UUID, vector: list[float]) -> None:
        key = f"user:{user_id}:embedding"
        await self.redis.set(key, _pack_embedding(vector), ex=self.settings.redis_embedding_ttl_seconds)

    async def get_user_embedding(self, user_id: UUID) -> list[float]:
        key = f"user:{user_id}:embedding"
        raw = await self.redis.get(key)
        if raw is None:
            raise FeatureNotFoundError(f"no cached embedding for user {user_id}")
        return _unpack_embedding(raw)

    # ---- recent interactions (session-aware re-ranking) ----------------
    async def add_recent_item(self, user_id: UUID, item_id: UUID, timestamp: float | None = None) -> None:
        key = f"user:{user_id}:recent"
        score = timestamp or time.time()
        await self.redis.zadd(key, {str(item_id): score})
        await self.redis.expire(key, self.settings.redis_recent_items_ttl_seconds)
        # keep only the last 50 interactions per the blueprint's sequence length
        await self.redis.zremrangebyrank(key, 0, -51)

    async def get_recent_items(self, user_id: UUID, limit: int = 50) -> list[str]:
        key = f"user:{user_id}:recent"
        return await self.redis.zrevrange(key, 0, limit - 1)

    # ---- item freshness / popularity composite --------------------------
    async def set_item_score(self, item_id: UUID, score: float) -> None:
        key = f"item:{item_id}:score"
        await self.redis.set(key, str(score), ex=self.settings.redis_freshness_ttl_seconds)

    async def get_item_score(self, item_id: UUID) -> float:
        key = f"item:{item_id}:score"
        raw = await self.redis.get(key)
        return float(raw) if raw is not None else 0.0

    async def batch_get_item_features(self, item_ids: list[UUID]) -> dict[str, float]:
        """Single-round-trip pipelined fetch — this is the ~8ms budget line
        item in the feed endpoint hot path (see blueprint's feed code sample).
        """
        if not item_ids:
            return {}
        pipe = self.redis.pipeline()
        for item_id in item_ids:
            pipe.get(f"item:{item_id}:score")
        results = await pipe.execute()
        return {
            str(item_id): (float(val) if val is not None else 0.0)
            for item_id, val in zip(item_ids, results, strict=True)
        }

    # ---- trending (real-time popularity, Addition 1) ---------------------
    async def incr_trending(self, item_id: UUID, by: float = 1.0) -> None:
        await self.redis.zincrby("trending:global", by, str(item_id))
        await self.redis.expire("trending:global", self.settings.redis_trending_ttl_seconds)

    async def get_trending(self, limit: int = 50) -> list[tuple[str, float]]:
        raw = await self.redis.zrevrange("trending:global", 0, limit - 1, withscores=True)
        return [(item_id, score) for item_id, score in raw]

    # ---- experiment assignment cache -------------------------------------
    async def cache_experiment_bucket(self, user_id: UUID, bucket: int) -> None:
        key = f"user:{user_id}:experiment"
        await self.redis.set(key, str(bucket), ex=self.settings.redis_experiment_ttl_seconds)

    async def get_cached_experiment_bucket(self, user_id: UUID) -> int | None:
        key = f"user:{user_id}:experiment"
        raw = await self.redis.get(key)
        return int(raw) if raw is not None else None

    # ---- HyperLogLog item popularity cardinality --------------------------
    async def track_unique_viewer(self, item_id: UUID, user_id: UUID) -> None:
        await self.redis.pfadd(f"item:{item_id}:unique_viewers", str(user_id))

    async def unique_viewer_count(self, item_id: UUID) -> int:
        return await self.redis.pfcount(f"item:{item_id}:unique_viewers")

    # ---- deterministic rate limiting (sliding window) ----------------------
    async def rate_limit_hit(self, api_key: str, window_seconds: int, limit: int) -> tuple[bool, int]:
        window = int(time.time() // window_seconds)
        key = f"ratelimit:{api_key}:{window}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, window_seconds)
        return count <= limit, count
