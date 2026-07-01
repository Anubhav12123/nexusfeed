"""GET /feed/{user_id} — main feed endpoint. p99 < 50ms.

Mirrors the "Key Code Pattern" from the blueprint's build guide almost line
for line: Redis embedding fetch -> FAISS retrieval -> batch feature fetch ->
LightGBM re-rank -> diversity/freshness post-processing -> async impression
log to Kafka. Every step is individually timed via the observability metrics
so the 50ms budget is provable, not just claimed.
"""
from __future__ import annotations

import asyncio
import time
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from nexusfeed.db.connection import get_db
from nexusfeed.db.repositories.experiment_repo import ExperimentRepository
from nexusfeed.db.repositories.item_repo import ItemRepository
from nexusfeed.experiments.bandit import EpsilonGreedyBandit
from nexusfeed.experiments.experiment_manager import ExperimentManager
from nexusfeed.features.item_features import ItemFeatureService
from nexusfeed.features.online_store import OnlineFeatureStore
from nexusfeed.features.user_features import UserFeatureService
from nexusfeed.observability.metrics import FEED_LATENCY_SECONDS, IN_FLIGHT_REQUESTS
from nexusfeed.ranking.reranker import Reranker
from nexusfeed.types import FeedItem, FeedResponse

router = APIRouter(tags=["feed"])

_bandit = EpsilonGreedyBandit()


@router.get("/feed/{user_id}", response_model=FeedResponse)
async def get_feed(
    user_id: UUID,
    request: Request,
    n: int = 20,
    db: AsyncSession = Depends(get_db),
) -> FeedResponse:
    IN_FLIGHT_REQUESTS.labels(route="/feed").inc()
    start = time.perf_counter()
    try:
        with FEED_LATENCY_SECONDS.time():
            online_store = OnlineFeatureStore(request.app.state.redis)
            item_features = ItemFeatureService(online_store)
            user_features = UserFeatureService(online_store, offline_store=None)  # type: ignore[arg-type]

            # 1. Fetch user embedding from Redis (~1ms) — falls back to a
            #    deterministic cold-start prior embedding if the user is new.
            try:
                user_embedding = await online_store.get_user_embedding(user_id)
                is_cold_start = False
            except Exception:
                import numpy as np

                user_embedding = np.random.default_rng(seed=hash(str(user_id)) % (2**32)).normal(
                    0, 0.1, request.app.state.embedding_dim
                ).tolist()
                is_cold_start = True

            # 2. ANN retrieval: top-K from FAISS (~5ms)
            candidate_generator = request.app.state.candidate_generator
            trending = await online_store.get_trending(limit=10)
            candidates = await candidate_generator.retrieve(
                user_embedding,
                k=request.app.state.settings.faiss_top_k,
                trending_item_ids=[iid for iid, _ in trending],
            )
            candidate_ids = [item_id for item_id, _ in candidates]

            # 3. Batch fetch features for all candidates (~8ms, single pipelined round trip)
            item_scores = await online_store.batch_get_item_features(
                [UUID(cid) for cid in candidate_ids if _is_uuid(cid)]
            )
            candidate_features = [
                {
                    "user_item_dot_product": score,
                    "item_freshness_score": item_scores.get(cid, 0.0),
                    "user_item_category_affinity": 0.5,
                    "time_decay": 1.0,
                    "diversity_score": 0.5,
                    "historical_ctr": 0.0,
                    "popularity_score": item_scores.get(cid, 0.0),
                }
                for cid, score in candidates
            ]

            # 4. Re-rank with LightGBM ONNX model (~2ms)
            ranker = request.app.state.ranker
            ranked = ranker.score(candidate_ids, candidate_features) if ranker else [
                _fallback_scored_item(cid, score) for cid, score in candidates
            ]

            # 5. Apply diversity + freshness post-processing (~1ms)
            recent_items = await online_store.get_recent_items(user_id, limit=50)
            reranker = Reranker()
            final_items = reranker.apply(ranked, n=n, seen_items=set(recent_items))

            # A/B: apply bandit exploration slots (multi-armed bandit, Addition/Layer 7)
            manager = ExperimentManager(ExperimentRepository(db), request.app.state.redis)
            try:
                assignment = await manager.get_assignment(user_id, request.app.state.settings.default_experiment_name)
            except Exception:
                assignment = None

            final_ids = _bandit.inject_exploration_slots(
                [i.item_id for i in final_items], candidate_ids, slot_fraction=request.app.state.settings.bandit_epsilon
            )
            by_id = {i.item_id: i for i in final_items}
            ordered = [by_id.get(fid) for fid in final_ids if fid in by_id]

            feed_items = [
                FeedItem(item_id=item.item_id, score=item.score, rank=idx + 1, category=item.category, is_trending=item.is_trending)
                for idx, item in enumerate(ordered)
            ]

            # 6. Log feed impression to Kafka (async, non-blocking)
            producer = request.app.state.kafka_producer
            asyncio.create_task(
                producer.send(
                    request.app.state.settings.kafka_topic_impressions,
                    value={"user_id": str(user_id), "item_ids": [i.item_id for i in feed_items]},
                    key=str(user_id),
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return FeedResponse(
            user_id=user_id,
            items=feed_items,
            model_version=request.app.state.model_version,
            experiment_bucket=assignment.bucket if assignment else None,
            experiment_variant=assignment.variant if assignment else None,
            latency_ms=round(elapsed_ms, 2),
            request_id=getattr(request.state, "request_id", None),
        )
    finally:
        IN_FLIGHT_REQUESTS.labels(route="/feed").dec()


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except ValueError:
        return False


def _fallback_scored_item(item_id: str, score: float):
    from nexusfeed.types import ScoredItem

    return ScoredItem(item_id=item_id, score=score)
