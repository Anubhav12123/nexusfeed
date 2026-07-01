"""Fixtures: mock Kafka, Redis, DB, sample events."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import numpy as np
import pytest

from nexusfeed.types import Event, EventType


@pytest.fixture
def sample_user_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def sample_item_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def sample_event(sample_user_id, sample_item_id) -> Event:
    return Event(user_id=sample_user_id, item_id=sample_item_id, event_type=EventType.CLICK, dwell_ms=3000)


@pytest.fixture
def mock_redis() -> AsyncMock:
    redis = AsyncMock()
    redis.pipeline.return_value = AsyncMock()
    return redis


@pytest.fixture
def mock_kafka_producer() -> AsyncMock:
    producer = AsyncMock()
    return producer


@pytest.fixture
def random_embeddings():
    def _make(n: int, dim: int = 128) -> np.ndarray:
        rng = np.random.default_rng(42)
        vectors = rng.normal(0, 1, size=(n, dim)).astype(np.float32)
        return vectors

    return _make
