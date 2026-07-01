"""Schema validation, deduplication, and anomaly detection for inbound events."""
from __future__ import annotations

import logging
import time

from redis.asyncio import Redis

from nexusfeed.exceptions import DuplicateEventError, EventValidationError
from nexusfeed.types import Event

logger = logging.getLogger(__name__)

MAX_CLOCK_SKEW_SECONDS = 300  # reject events claiming to be >5 min in the future
MAX_EVENT_AGE_SECONDS = 7 * 24 * 3600  # reject events older than the Kafka retention window
DEDUPE_KEY_TTL_SECONDS = 3600


class EventValidator:
    """Rejects malformed, duplicate, or anomalous events before they hit Kafka."""

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def validate(self, event: Event) -> None:
        self._check_timestamp(event)
        await self._check_duplicate(event)

    def _check_timestamp(self, event: Event) -> None:
        if event.client_timestamp is None:
            return
        now = time.time()
        ts = event.client_timestamp.timestamp()
        if ts > now + MAX_CLOCK_SKEW_SECONDS:
            raise EventValidationError(
                f"event {event.event_id} timestamp is {ts - now:.0f}s in the future (clock skew)"
            )
        if now - ts > MAX_EVENT_AGE_SECONDS:
            raise EventValidationError(
                f"event {event.event_id} is older than the retention window ({MAX_EVENT_AGE_SECONDS}s)"
            )

    async def _check_duplicate(self, event: Event) -> None:
        """SET-based dedupe: NX write returns False if the key already existed."""
        key = f"dedupe:event:{event.event_id}"
        was_set = await self.redis.set(key, "1", nx=True, ex=DEDUPE_KEY_TTL_SECONDS)
        if not was_set:
            raise DuplicateEventError(f"event {event.event_id} already processed")
