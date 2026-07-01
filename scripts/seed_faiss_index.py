"""Builds the FAISS index from item embeddings in Postgres, trains the
LightGBM ranking model on the synthetic interaction features, and warms the
Redis online feature store (user embeddings, item freshness/popularity,
trending) so the API is immediately servable without waiting on Kafka
traffic to accumulate.

Run after scripts/generate_synthetic_data.py:
    python -m scripts.seed_faiss_index
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
from redis.asyncio import Redis

# `import nexusfeed` (triggered by the first `nexusfeed.*` import below)
# forces lightgbm's OpenMP runtime to initialize before faiss's — must
# happen before `import faiss`, so this block stays above it.
import nexusfeed  # noqa: F401,E402

import faiss  # noqa: E402

# On macOS, the faiss-cpu and lightgbm pip wheels each bundle their own
# private libomp.dylib. Even with lightgbm's runtime initialized first,
# FAISS's HNSW `.add()` still spins up its OWN OpenMP thread pool for
# parallel graph construction, and two live OpenMP runtimes in one process
# segfaults regardless of import order (reproduced directly: IndexHNSWFlat
# .add() crashes on macOS host, but is fine inside the Linux Docker
# containers, which share one system libgomp). This script is the one place
# meant to run on a bare macOS host rather than in a container, so pin FAISS
# to single-threaded here — cheap for the thousands-of-items scale this
# script seeds, and irrelevant to the actual API/worker processes, which run
# in Docker and are unaffected.
faiss.omp_set_num_threads(1)

from nexusfeed.config import get_settings
from nexusfeed.db.connection import get_database
from nexusfeed.features.item_features import compute_composite_score, compute_freshness_score
from nexusfeed.models.ranking_model import RANKING_FEATURE_NAMES, RankingModel
from nexusfeed.retrieval.index_builder import IndexBuilder


async def build_faiss_index(settings) -> None:
    db = get_database()
    async with db.session_factory() as session:
        builder = IndexBuilder(session, embedding_dim=settings.embedding_dim)
        index = await builder.build_global_index()
    Path(settings.ann_index_path).parent.mkdir(parents=True, exist_ok=True)
    index.save(settings.ann_index_path)
    print(f"FAISS index built with {index.size} items -> {settings.ann_index_path}")
    await db.dispose()


def train_ranking_model(data_dir: str) -> RankingModel:
    df = pd.read_parquet(Path(data_dir) / "interactions.parquet")
    feature_df = pd.DataFrame(
        {
            "user_item_dot_product": df["user_item_dot_product"],
            "item_freshness_score": df["freshness_score"],
            "user_item_category_affinity": df["historical_ctr"],
            "time_decay": 1.0 - (df["time_of_day"].astype(float) / 24.0),
            "diversity_score": np.random.default_rng(0).uniform(0, 1, len(df)),
            "historical_ctr": df["historical_ctr"],
            "popularity_score": df["historical_ctr"],
        }
    )
    features = feature_df[RANKING_FEATURE_NAMES].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy()

    model = RankingModel()
    model.train(features, labels)
    output_path = Path(data_dir) / "ranking_model.txt"
    model.save(output_path)
    print(f"Ranking model trained on {len(df)} rows -> {output_path}")
    return model


async def warm_redis(settings, data_dir: str) -> None:
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    from nexusfeed.features.online_store import OnlineFeatureStore

    store = OnlineFeatureStore(redis, settings)

    item_ids = np.load(Path(data_dir) / "item_ids.npy", allow_pickle=True)
    item_factors = np.load(Path(data_dir) / "item_factors.npy")
    user_ids = np.load(Path(data_dir) / "user_ids.npy", allow_pickle=True)
    user_factors = np.load(Path(data_dir) / "user_factors.npy")

    from uuid import UUID

    for uid, factor in zip(user_ids[:500], user_factors[:500], strict=True):  # warm a sample for the demo
        await store.set_user_embedding(UUID(str(uid)), factor.tolist())

    rng = np.random.default_rng(1)
    for item_id, popularity in zip(item_ids, rng.uniform(0, 1, len(item_ids)), strict=True):
        freshness = compute_freshness_score(__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
        composite = compute_composite_score(freshness, float(popularity))
        await store.set_item_score(UUID(str(item_id)), composite)
        if popularity > 0.9:
            await store.incr_trending(UUID(str(item_id)), by=float(popularity) * 10)

    await redis.aclose()
    print(f"Warmed Redis with {min(500, len(user_ids))} user embeddings and {len(item_ids)} item scores.")


async def main() -> None:
    settings = get_settings()
    await build_faiss_index(settings)
    train_ranking_model(settings.__dict__.get("data_dir", "./data"))
    await warm_redis(settings, "./data")
    print("Seeding complete. Start the API and hit GET /feed/{user_id}.")


if __name__ == "__main__":
    asyncio.run(main())
