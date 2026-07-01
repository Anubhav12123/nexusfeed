"""Consumer group manager with manual offset commit and error handling.

Offsets are committed only after the message has been successfully applied to
downstream state (Redis feature update, Postgres audit write). This gives
at-least-once delivery: a crash between "process" and "commit" replays the
message rather than silently dropping it.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError
from aiokafka.structs import ConsumerRecord

from nexusfeed.config import Settings, get_settings

logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict], Awaitable[None]]


class KafkaEventConsumer:
    def __init__(
        self,
        topics: list[str],
        group_id: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.topics = topics
        self.group_id = group_id or self.settings.kafka_consumer_group
        self._consumer: AIOKafkaConsumer | None = None

    async def start(self) -> None:
        import json

        self._consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.settings.kafka_bootstrap_list,
            group_id=self.group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await self._consumer.start()
        logger.info("kafka_consumer_started", extra={"topics": self.topics, "group": self.group_id})

    async def stop(self) -> None:
        if self._consumer is not None:
            await self._consumer.stop()

    async def consume_forever(self, handler: MessageHandler, max_retries: int = 3) -> None:
        if self._consumer is None:
            raise RuntimeError("call start() before consume_forever()")

        async for record in self._consumer:
            await self._process_with_retry(record, handler, max_retries)

    async def _process_with_retry(
        self, record: ConsumerRecord, handler: MessageHandler, max_retries: int
    ) -> None:
        attempt = 0
        while True:
            try:
                await handler(record.value)
                await self._consumer.commit()
                return
            except KafkaError:
                raise
            except Exception as exc:  # noqa: BLE001 - deliberately broad: never crash the loop
                attempt += 1
                logger.error(
                    "consumer_handler_error",
                    extra={
                        "topic": record.topic,
                        "partition": record.partition,
                        "offset": record.offset,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
                if attempt > max_retries:
                    # Give up on this message but still commit forward, otherwise a
                    # single poison message stalls the whole partition forever.
                    logger.error(
                        "consumer_handler_giving_up",
                        extra={"topic": record.topic, "offset": record.offset},
                    )
                    await self._consumer.commit()
                    return
