import uuid

from nexusfeed.types import Event, EventType, ItemProfile, UserProfile


def test_signal_weight_matches_event_table():
    event = Event(user_id=uuid.uuid4(), item_id=uuid.uuid4(), event_type=EventType.SHARE)
    assert event.signal_weight() == 3.0


def test_negative_events_flagged():
    event = Event(user_id=uuid.uuid4(), item_id=uuid.uuid4(), event_type=EventType.SKIP)
    assert event.is_negative() is True

    positive = Event(user_id=uuid.uuid4(), item_id=uuid.uuid4(), event_type=EventType.CLICK)
    assert positive.is_negative() is False


def test_user_cold_start_threshold():
    user = UserProfile(user_id=uuid.uuid4(), interaction_count=3)
    assert user.is_cold_start() is True
    user.interaction_count = 20
    assert user.is_cold_start() is False


def test_item_cold_start_threshold():
    item = ItemProfile(item_id=uuid.uuid4(), category="tech", impression_count=5)
    assert item.is_cold_start() is True
    item.impression_count = 500
    assert item.is_cold_start() is False
