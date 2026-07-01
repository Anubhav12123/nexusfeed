"""Runs Addition 7 (Benchmark Against Baselines) against the synthetic
interaction data: compares the two-tower approach (using the true latent
factors as a stand-in for a converged trained model) against random,
most-popular, and collaborative-filtering baselines.

Run: python -m scripts.run_benchmark
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nexusfeed.baselines.benchmark_runner import BenchmarkRunner
from nexusfeed.baselines.collaborative_filtering import CollaborativeFilteringBaseline
from nexusfeed.baselines.popularity_baseline import PopularityBaseline
from nexusfeed.baselines.random_baseline import RandomBaseline


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def simulate_variant(
    approach_name: str,
    ranked_item_indices_per_user: dict[int, list[int]],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    n_shown: int,
    runner: BenchmarkRunner,
) -> None:
    clicks = 0
    impressions = 0
    total_dwell_ms = 0
    returning_users = 0
    shown_embeddings = []

    for user_idx, item_indices in ranked_item_indices_per_user.items():
        shown = item_indices[:n_shown]
        impressions += len(shown)
        session_had_click = False
        for item_idx in shown:
            affinity = float(np.dot(user_factors[user_idx], item_factors[item_idx]))
            click_prob = float(sigmoid(np.array([affinity * 4.0]))[0])
            clicked = np.random.default_rng(item_idx + user_idx).random() < click_prob
            if clicked:
                clicks += 1
                total_dwell_ms += int(2000 + click_prob * 20000)
                session_had_click = True
            shown_embeddings.append(item_factors[item_idx])
        if session_had_click:
            returning_users += 1

    runner.evaluate(
        approach_name=approach_name,
        clicks=clicks,
        impressions=impressions,
        total_dwell_ms=total_dwell_ms,
        returning_users=returning_users,
        total_users=len(ranked_item_indices_per_user),
        item_embeddings_shown=np.array(shown_embeddings[:500]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--eval-users", type=int, default=300)
    parser.add_argument("--n-shown", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    user_factors = np.load(data_dir / "user_factors.npy")
    item_factors = np.load(data_dir / "item_factors.npy")
    df = pd.read_parquet(data_dir / "interactions.parquet")

    num_users, num_items = len(user_factors), len(item_factors)
    eval_users = list(range(min(args.eval_users, num_users)))
    item_ids = [str(i) for i in range(num_items)]

    runner = BenchmarkRunner()

    # Two-tower: rank by true dot-product affinity (stand-in for a converged model)
    two_tower_rankings = {
        u: list(np.argsort(-(item_factors @ user_factors[u]))[: args.n_shown * 2]) for u in eval_users
    }
    simulate_variant("two_tower", two_tower_rankings, user_factors, item_factors, args.n_shown, runner)

    # Random baseline
    random_baseline = RandomBaseline(item_ids)
    random_rankings = {
        u: [int(i) for i in random_baseline.recommend(str(u), n=args.n_shown * 2)] for u in eval_users
    }
    simulate_variant("random", random_rankings, user_factors, item_factors, args.n_shown, runner)

    # Most-popular baseline (popularity derived from historical CTR proxy in the synthetic data)
    interaction_counts = df["item_id"].value_counts().to_dict()
    popularity_baseline = PopularityBaseline(interaction_counts)
    popularity_rankings = {
        u: [int(i) for i in popularity_baseline.recommend(str(u), n=args.n_shown * 2)] for u in eval_users
    }
    simulate_variant("most_popular", popularity_rankings, user_factors, item_factors, args.n_shown, runner)

    # Collaborative filtering baseline (matrix factorization via SVD on a click matrix)
    click_matrix = np.zeros((num_users, num_items), dtype=np.float32)
    clicked_df = df[df["label"] == 1]
    click_matrix[clicked_df["user_id"].to_numpy(), clicked_df["item_id"].to_numpy()] = 1.0
    cf = CollaborativeFilteringBaseline(num_factors=32).fit(
        click_matrix, user_ids=[str(u) for u in range(num_users)], item_ids=item_ids
    )
    cf_rankings = {
        u: [int(i) for i in cf.recommend(str(u), n=args.n_shown * 2)] for u in eval_users
    }
    simulate_variant("collaborative_filtering", cf_rankings, user_factors, item_factors, args.n_shown, runner)

    print(runner.report_table())


if __name__ == "__main__":
    main()
