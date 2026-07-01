"""Routes validated events to the correct Kafka topic based on event type."""
from __future__ import annotations

from nexusfeed.config import Settings, get_settings
from nexusfeed.ingestion.kafka_producer import KafkaEventProducer
from nexusfeed.types import Event


class EventRouter:
    """user-events topic gets everything (partitioned by user_id for ordered
    per-user processing); item-events is a fan-out copy keyed by item_id so the
    item-popularity aggregator can consume independently without contending
    with the per-user feature pipeline consumer group.
    """

    def __init__(self, producer: KafkaEventProducer, settings: Settings | None = None) -> None:
        self.producer = producer
        self.settings = settings or get_settings()

    async def route(self, event: Event) -> None:
        payload = event.model_dump(mode="json")
        await self.producer.send(
            self.settings.kafka_topic_user_events,
            value=payload,
            key=str(event.user_id),
        )
        await self.producer.send(
            self.settings.kafka_topic_item_events,
            value=payload,
            key=str(event.item_id),
        )
