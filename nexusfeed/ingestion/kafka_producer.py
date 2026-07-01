"""Async Kafka producer with batching and retry logic.

Fire-and-forget semantics for POST /events: the caller gets an ack as soon as
Kafka has the message queued for the leader, not after downstream processing.
batch_size / linger_ms are tuned for throughput over per-message latency —
see NexusFeed_Project_Blueprint.pdf Layer 1 for the Kafka-vs-RabbitMQ rationale.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from nexusfeed.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _json_serializer(value: dict[str, Any]) -> bytes:
    return json.dumps(value, default=str).encode("utf-8")


class KafkaEventProducer:
    """Thin async wrapper around aiokafka with exponential-backoff retry."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.settings.kafka_bootstrap_list,
            value_serializer=_json_serializer,
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            max_batch_size=self.settings.kafka_batch_size,
            linger_ms=self.settings.kafka_linger_ms,
            acks="all",
            enable_idempotence=True,
        )
        await self._producer.start()
        logger.info("kafka_producer_started", extra={"brokers": self.settings.kafka_bootstrap_list})

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()

    async def send(
        self,
        topic: str,
        value: dict[str, Any],
        key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        """Send with exponential backoff retry on retriable Kafka errors
        (e.g. LEADER_NOT_AVAILABLE during in-flight partition leader election).
        """
        if self._producer is None:
            raise RuntimeError("KafkaEventProducer.start() must be called before send()")

        attempt = 0
        while True:
            try:
                await self._producer.send_and_wait(topic, value=value, key=key)
                return
            except KafkaError as exc:
                attempt += 1
                if attempt > max_retries:
                    logger.error("kafka_send_failed", extra={"topic": topic, "error": str(exc)})
                    raise
                backoff = min(0.05 * (2**attempt), 2.0)
                logger.warning(
                    "kafka_send_retry",
                    extra={"topic": topic, "attempt": attempt, "backoff_s": backoff, "error": str(exc)},
                )
                await asyncio.sleep(backoff)

    def send_nowait(self, topic: str, value: dict[str, Any], key: str | None = None) -> None:
        """Non-blocking fire-and-forget helper for hot paths (e.g. impression logging)."""
        asyncio.create_task(self.send(topic, value, key))


_producer_singleton: KafkaEventProducer | None = None


async def get_kafka_producer() -> KafkaEventProducer:
    global _producer_singleton
    if _producer_singleton is None:
        _producer_singleton = KafkaEventProducer()
        await _producer_singleton.start()
    return _producer_singleton
