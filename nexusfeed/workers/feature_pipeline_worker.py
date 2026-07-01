"""Standalone consumer-group worker process — runs the Flink-job-equivalent
FeaturePipeline against Kafka's user-events/item-events topics. Deployed as
its own Kubernetes Deployment (see infra/kubernetes/worker-deployment.yaml)
so it scales independently of the API pods, matching the blueprint's
Dockerfile.worker split.
"""
from __future__ import annotations

import asyncio
import logging

from redis.asyncio import Redis

from nexusfeed.config import get_settings
from nexusfeed.features.item_features import ItemFeatureService
from nexusfeed.features.feature_pipeline import FeaturePipeline
from nexusfeed.features.online_store import OnlineFeatureStore
from nexusfeed.ingestion.kafka_consumer import KafkaEventConsumer
from nexusfeed.observability.logging import configure_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_format=(settings.log_format == "json"))

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    online_store = OnlineFeatureStore(redis, settings)
    item_features = ItemFeatureService(online_store)
    pipeline = FeaturePipeline(online_store, item_features)

    consumer = KafkaEventConsumer(
        topics=[settings.kafka_topic_user_events, settings.kafka_topic_item_events],
        group_id=settings.kafka_consumer_group,
    )
    await consumer.start()
    logger.info("feature_pipeline_worker_started")

    try:
        await pipeline.run(consumer)
    finally:
        await consumer.stop()
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
