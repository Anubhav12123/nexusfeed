"""Real-time feature computation stream job.

In production this is a Flink job re-computing user embeddings every 5
minutes from the Kafka event stream. Locally, `FeaturePipelineConsumer` plays
the same role as a plain aiokafka consumer group member: it is horizontally
scalable (add more instances = more partitions consumed) and stateless
between restarts (all state lives in Redis/Postgres, not in-process).
"""
from __future__ import annotations

import logging
from uuid import UUID

from nexusfeed.features.item_features import ItemFeatureService
from nexusfeed.features.online_store import OnlineFeatureStore
from nexusfeed.ingestion.kafka_consumer import KafkaEventConsumer
from nexusfeed.observability.metrics import EVENTS_CONSUMED, FEATURE_PIPELINE_LAG_SECONDS
from nexusfeed.types import Event

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Consumes user-events / item-events and updates the online feature store
    incrementally (Redis sorted-set append, trending zset incr) so serving
    always reads fresh-enough features without waiting for the batch job.
    """

    def __init__(self, online_store: OnlineFeatureStore, item_features: ItemFeatureService) -> None:
        self.online_store = online_store
        self.item_features = item_features

    async def handle_event(self, raw: dict) -> None:
        event = Event.model_validate(raw)
        EVENTS_CONSUMED.labels(event_type=event.event_type.value).inc()

        await self.online_store.add_recent_item(UUID(str(event.user_id)), UUID(str(event.item_id)))
        await self.online_store.track_unique_viewer(UUID(str(event.item_id)), UUID(str(event.user_id)))

        weight = event.signal_weight()
        if weight > 0:
            await self.online_store.incr_trending(UUID(str(event.item_id)), by=weight)

        lag = 0.0
        if event.created_at:
            import time

            lag = max(time.time() - event.created_at.timestamp(), 0.0)
        FEATURE_PIPELINE_LAG_SECONDS.set(lag)

    async def run(self, consumer: KafkaEventConsumer) -> None:
        await consumer.consume_forever(self.handle_event)
