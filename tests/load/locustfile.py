"""Load test: target 1000 RPS sustained, p99 < 50ms on GET /feed.

Run: locust -f tests/load/locustfile.py --headless -u 1000 -r 50 -t 10m --host http://localhost:8000
(blueprint Phase 7 milestone: 1000 RPS sustained for 10 minutes, p99 < 50ms)
"""
from __future__ import annotations

import random
import uuid

from locust import HttpUser, between, task

from nexusfeed.config import get_settings


class NexusFeedUser(HttpUser):
    wait_time = between(0.01, 0.05)

    def on_start(self) -> None:
        # Reads VALID_API_KEYS from the same .env the target API uses,
        # rather than a literal that goes stale the moment the key rotates.
        api_key = next(iter(get_settings().api_keys))
        self.headers = {"X-API-Key": api_key}
        self.user_id = str(uuid.uuid4())

    @task(10)
    def get_feed(self) -> None:
        self.client.get(
            f"/feed/{self.user_id}", params={"n": 20}, headers=self.headers, name="/feed/[user_id]"
        )

    @task(5)
    def post_event(self) -> None:
        payload = {
            "user_id": self.user_id,
            "item_id": str(uuid.uuid4()),
            "event_type": random.choice(["click", "view_full", "skip", "share"]),
            "dwell_ms": random.randint(500, 30000),
        }
        self.client.post("/events", json=payload, headers=self.headers, name="/events")

    @task(2)
    def get_similar(self) -> None:
        self.client.get(
            f"/similar/{uuid.uuid4()}", params={"n": 10}, headers=self.headers, name="/similar/[item_id]"
        )

    @task(1)
    def get_health(self) -> None:
        self.client.get("/health", name="/health")
