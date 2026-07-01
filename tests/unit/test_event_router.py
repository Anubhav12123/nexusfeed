import uuid
from unittest.mock import AsyncMock

import pytest

from nexusfeed.config import Settings
from nexusfeed.ingestion.event_router import EventRouter
from nexusfeed.types import Event, EventType


@pytest.mark.asyncio
async def test_route_sends_to_both_user_and_item_topics():
    producer = AsyncMock()
    settings = Settings()
    router = EventRouter(producer, settings)
    event = Event(user_id=uuid.uuid4(), item_id=uuid.uuid4(), event_type=EventType.CLICK)

    await router.route(event)

    assert producer.send.call_count == 2
    topics_called = {call.args[0] for call in producer.send.call_args_list}
    assert topics_called == {settings.kafka_topic_user_events, settings.kafka_topic_item_events}


@pytest.mark.asyncio
async def test_route_keys_by_user_id_and_item_id_respectively():
    producer = AsyncMock()
    settings = Settings()
    router = EventRouter(producer, settings)
    user_id, item_id = uuid.uuid4(), uuid.uuid4()
    event = Event(user_id=user_id, item_id=item_id, event_type=EventType.SHARE)

    await router.route(event)

    keys_by_topic = {call.args[0]: call.kwargs["key"] for call in producer.send.call_args_list}
    assert keys_by_topic[settings.kafka_topic_user_events] == str(user_id)
    assert keys_by_topic[settings.kafka_topic_item_events] == str(item_id)
