"""Walks through the full NexusFeed request flow against a running API:
ingest an event -> fetch a personalized feed -> look up similar items ->
check the A/B experiment assignment -> pull an explanation -> peek at
trending items. Useful as a smoke test and as the script behind a live demo.

Run: python -m scripts.demo --user-id <uuid> --item-id <uuid>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import UUID

import httpx
import numpy as np

from nexusfeed.config import get_settings


def _pretty(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, default=str)[:2000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--item-id", default=None)
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    if args.user_id is None or args.item_id is None:
        user_ids = np.load(Path(args.data_dir) / "user_ids.npy", allow_pickle=True)
        item_ids = np.load(Path(args.data_dir) / "item_ids.npy", allow_pickle=True)
        args.user_id = args.user_id or str(user_ids[0])
        args.item_id = args.item_id or str(item_ids[0])

    # Reads the same VALID_API_KEYS the running API was started with (from
    # .env), rather than a literal that would go stale the moment you rotate
    # the key in one place and not the other.
    api_key = next(iter(get_settings().api_keys))
    headers = {"X-API-Key": api_key}
    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=10.0)

    health = client.get("/health").json()
    _pretty("Health", health)

    event = {
        "user_id": args.user_id,
        "item_id": args.item_id,
        "event_type": "click",
        "dwell_ms": 4200,
    }
    ingest = client.post("/events", json=event).json()
    _pretty("Event ingested", ingest)

    feed = client.get(f"/feed/{args.user_id}", params={"n": 10}).json()
    _pretty(f"Feed for user {args.user_id}", feed)

    similar = client.get(f"/similar/{args.item_id}", params={"n": 5}).json()
    _pretty(f"Items similar to {args.item_id}", similar)

    experiment = client.get(f"/experiments/{args.user_id}").json()
    _pretty("Experiment assignment", experiment)

    if feed.get("items"):
        top_item = feed["items"][0]["item_id"]
        explanation = client.get(f"/explain/{args.user_id}/{top_item}").json()
        _pretty(f"Why item {top_item} was recommended", explanation)

    trending = client.get("/admin/trending", params={"limit": 5}).json()
    _pretty("Trending items", trending)

    status = client.get("/admin/system-status").json()
    _pretty("System status", status)

    print(f"\nFeed latency reported: {feed.get('latency_ms', 'n/a')} ms (target: p99 < 50ms)")


if __name__ == "__main__":
    main()
