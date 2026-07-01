"""Generates a synthetic-but-structured dataset so the whole pipeline is
demoable end to end without needing real user traffic:

- Users and items get latent preference/content factors (this stands in for
  what the two-tower model would eventually learn).
- Interactions are sampled with click probability = sigmoid(dot(user, item))
  plus category-affinity noise, so there is genuine learnable signal.
- Items are written to Postgres with their embedding pre-populated (pgvector)
  so retrieval and ranking work immediately; interactions are written both to
  Postgres (audit trail) and to a local Parquet file (training input).

Run: python -m scripts.generate_synthetic_data --users 2000 --items 5000 --interactions 50000
"""
from __future__ import annotations

import argparse
import asyncio
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nexusfeed.config import get_settings
from nexusfeed.db.connection import get_database
from nexusfeed.db.models import Base, Interaction, Item, User

CATEGORIES = ["news", "sports", "tech", "music", "gaming", "food", "travel", "fashion", "finance", "comedy"]
DEVICES = ["ios", "android", "web", "tv"]
EMBEDDING_DIM = 128
CONTENT_EMBEDDING_DIM = 768


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_users(n: int, rng: np.random.Generator) -> tuple[list[dict], np.ndarray]:
    factors = rng.normal(0, 1, size=(n, EMBEDDING_DIM))
    factors /= np.linalg.norm(factors, axis=1, keepdims=True)
    users = [
        {
            "device": random.choice(DEVICES),
            "onboarding_categories": random.sample(CATEGORIES, k=random.randint(1, 3)),
        }
        for _ in range(n)
    ]
    return users, factors


def generate_items(n: int, rng: np.random.Generator) -> tuple[list[dict], np.ndarray]:
    factors = rng.normal(0, 1, size=(n, EMBEDDING_DIM))
    factors /= np.linalg.norm(factors, axis=1, keepdims=True)
    now = datetime.now(timezone.utc)
    items = [
        {
            "category": random.choice(CATEGORIES),
            "content_text": f"synthetic item {i} about {random.choice(CATEGORIES)}",
            "created_at": now - timedelta(hours=random.randint(0, 24 * 60)),
        }
        for i in range(n)
    ]
    return items, factors


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--items", type=int, default=5000)
    parser.add_argument("--interactions", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./data")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.users} users, {args.items} items, {args.interactions} interactions...")
    user_meta, user_factors = generate_users(args.users, rng)
    item_meta, item_factors = generate_items(args.items, rng)

    db = get_database()
    async with db.engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: None)  # ensure connection works; migrations create schema

    user_ids: list[str] = []
    item_ids: list[str] = []

    async with db.session_factory() as session:
        users = [User(metadata_={"device": m["device"], "onboarding_categories": m["onboarding_categories"]}) for m in user_meta]
        session.add_all(users)
        await session.flush()
        user_ids = [str(u.id) for u in users]

        items = [
            Item(
                category=m["category"],
                content_text=m["content_text"],
                embedding=item_factors[i].tolist(),
                created_at=m["created_at"],
            )
            for i, m in enumerate(item_meta)
        ]
        session.add_all(items)
        await session.flush()
        item_ids = [str(it.id) for it in items]

        await session.commit()
    print(f"Wrote {len(user_ids)} users and {len(item_ids)} items to Postgres.")

    rows = []
    interactions_to_insert = []
    for _ in range(args.interactions):
        ui = random.randrange(args.users)
        ii = random.randrange(args.items)
        affinity = float(np.dot(user_factors[ui], item_factors[ii]))
        click_prob = float(sigmoid(np.array([affinity * 4.0]))[0])
        clicked = random.random() < click_prob
        event_type = "click" if clicked else random.choice(["view_full", "view_scroll_past", "skip"])
        dwell_ms = random.randint(500, 60000) if clicked else None

        created_at = datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 60 * 24 * 30))
        interactions_to_insert.append(
            Interaction(
                user_id=user_ids[ui],
                item_id=item_ids[ii],
                event_type=event_type,
                dwell_ms=dwell_ms,
                created_at=created_at,
            )
        )

        rows.append(
            {
                "user_id": ui,
                "item_id": ii,
                "category_id": CATEGORIES.index(item_meta[ii]["category"]),
                "device_id": DEVICES.index(user_meta[ui]["device"]),
                "time_of_day": created_at.hour,
                "age": float(random.randint(13, 65)),
                "label": int(clicked),
                "recent_item_sequence": [random.randrange(args.items) for _ in range(10)],
                "content_embedding": rng.normal(0, 1, CONTENT_EMBEDDING_DIM).tolist(),
                "freshness_score": max(0.0, 1 - (created_at - item_meta[ii]["created_at"]).days / 60),
                "historical_ctr": click_prob,
                "user_item_dot_product": affinity,
            }
        )

    async with db.session_factory() as session:
        batch_size = 1000
        for i in range(0, len(interactions_to_insert), batch_size):
            session.add_all(interactions_to_insert[i : i + batch_size])
            await session.flush()
        await session.commit()
    print(f"Wrote {len(interactions_to_insert)} interactions to Postgres.")

    df = pd.DataFrame(rows)
    parquet_path = Path(args.output_dir) / "interactions.parquet"
    df.to_parquet(parquet_path)
    print(f"Wrote training dataframe to {parquet_path}")

    np.save(Path(args.output_dir) / "user_ids.npy", np.array(user_ids))
    np.save(Path(args.output_dir) / "item_ids.npy", np.array(item_ids))
    np.save(Path(args.output_dir) / "user_factors.npy", user_factors)
    np.save(Path(args.output_dir) / "item_factors.npy", item_factors)

    await db.dispose()
    print("Done. Next: python -m scripts.seed_faiss_index")


if __name__ == "__main__":
    asyncio.run(main())
