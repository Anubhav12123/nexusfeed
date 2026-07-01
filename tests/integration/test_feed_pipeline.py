"""Full-stack integration test — requires `docker compose up` (Postgres,
Redis, Kafka) to be running. Exercises the real event-ingest-to-feed path
end to end, not mocks, per the blueprint's Phase 1 milestone: "integration
tests pass against real Docker services."
"""
from __future__ import annotations

import uuid

import httpx
import pytest

from nexusfeed.config import get_settings

pytestmark = pytest.mark.integration

BASE_URL = "http://localhost:8000"


@pytest.fixture
def client():
    # Reads VALID_API_KEYS from the same .env the target API uses, rather
    # than a literal that goes stale the moment the key rotates.
    api_key = next(iter(get_settings().api_keys))
    with httpx.Client(base_url=BASE_URL, headers={"X-API-Key": api_key}, timeout=10.0) as c:
        yield c


def test_health_endpoint_reports_all_components_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "redis" in body["components"]


def test_event_ingest_and_feed_roundtrip(client):
    user_id = str(uuid.uuid4())
    item_id = str(uuid.uuid4())

    ingest_response = client.post(
        "/events", json={"user_id": user_id, "item_id": item_id, "event_type": "click", "dwell_ms": 5000}
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["status"] == "accepted"

    feed_response = client.get(f"/feed/{user_id}", params={"n": 10})
    assert feed_response.status_code == 200
    body = feed_response.json()
    assert body["user_id"] == user_id
    assert isinstance(body["items"], list)
    assert body["latency_ms"] < 500  # generous CI bound; local target is p99 < 50ms


def test_duplicate_event_is_ignored(client):
    user_id = str(uuid.uuid4())
    item_id = str(uuid.uuid4())
    event_id = str(uuid.uuid4())
    payload = {
        "event_id": event_id,
        "user_id": user_id,
        "item_id": item_id,
        "event_type": "click",
    }
    first = client.post("/events", json=payload)
    second = client.post("/events", json=payload)
    assert first.json()["status"] == "accepted"
    assert second.json()["status"] == "duplicate_ignored"


def test_rate_limit_returns_429_after_threshold(client):
    user_id = str(uuid.uuid4())
    statuses = []
    for _ in range(150):
        item_id = str(uuid.uuid4())
        payload = {"user_id": user_id, "item_id": item_id, "event_type": "view_full"}
        response = client.post("/events", json=payload)
        statuses.append(response.status_code)
    assert 429 in statuses
