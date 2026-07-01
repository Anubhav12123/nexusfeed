import uuid
from datetime import datetime, timedelta, timezone

import fakeredis.aioredis
import pytest

from nexusfeed.exceptions import DuplicateEventError, EventValidationError
from nexusfeed.ingestion.event_validator import EventValidator
from nexusfeed.types import Event, EventType


@pytest.fixture
async def validator():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield EventValidator(redis)
    await redis.aclose()


def _event(**overrides):
    defaults = dict(user_id=uuid.uuid4(), item_id=uuid.uuid4(), event_type=EventType.CLICK)
    defaults.update(overrides)
    return Event(**defaults)


@pytest.mark.asyncio
async def test_valid_event_passes(validator):
    await validator.validate(_event())  # should not raise


@pytest.mark.asyncio
async def test_duplicate_event_rejected_on_second_call(validator):
    event = _event()
    await validator.validate(event)
    with pytest.raises(DuplicateEventError):
        await validator.validate(event)


@pytest.mark.asyncio
async def test_future_timestamp_rejected(validator):
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    with pytest.raises(EventValidationError):
        await validator.validate(_event(client_timestamp=future))


@pytest.mark.asyncio
async def test_ancient_timestamp_rejected(validator):
    ancient = datetime.now(timezone.utc) - timedelta(days=30)
    with pytest.raises(EventValidationError):
        await validator.validate(_event(client_timestamp=ancient))


@pytest.mark.asyncio
async def test_recent_timestamp_accepted(validator):
    recent = datetime.now(timezone.utc) - timedelta(minutes=5)
    await validator.validate(_event(client_timestamp=recent))  # should not raise
