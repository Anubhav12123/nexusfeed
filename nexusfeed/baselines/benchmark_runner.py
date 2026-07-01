"""Addition 7 — Benchmark Against Baselines (the research signal).

Runs a controlled comparison of NexusFeed's two-tower model against three
baselines (random, most-popular, collaborative filtering) on CTR, dwell
time, return rate, and diversity, and formats the result as the public
technical-report table referenced in the blueprint's interview answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BenchmarkResult:
    approach_name: str
    ctr: float
    avg_dwell_ms: float
    return_rate: float
    diversity_score: float

    def ctr_lift_vs(self, other: "BenchmarkResult") -> float:
        return (self.ctr - other.ctr) / other.ctr if other.ctr > 0 else float("inf")


class BenchmarkRunner:
    """Simulates or replays interaction logs against each candidate ranker
    and aggregates the four headline metrics. In a real 30-day study this
    consumes actual logged impressions/clicks per variant; for local/demo
    use it can run against synthetically generated interaction logs from
    scripts/generate_synthetic_data.py.
    """

    def __init__(self) -> None:
        self.results: list[BenchmarkResult] = []

    def evaluate(
        self,
        approach_name: str,
        clicks: int,
        impressions: int,
        total_dwell_ms: int,
        returning_users: int,
        total_users: int,
        item_embeddings_shown: np.ndarray,
    ) -> BenchmarkResult:
        from nexusfeed.models.evaluate import intra_list_diversity

        result = BenchmarkResult(
            approach_name=approach_name,
            ctr=clicks / impressions if impressions else 0.0,
            avg_dwell_ms=total_dwell_ms / impressions if impressions else 0.0,
            return_rate=returning_users / total_users if total_users else 0.0,
            diversity_score=intra_list_diversity(item_embeddings_shown),
        )
        self.results.append(result)
        return result

    def report_table(self) -> str:
        header = f"{'Approach':<25}{'CTR':>10}{'Avg Dwell (ms)':>18}{'Return Rate':>15}{'Diversity':>12}"
        lines = [header, "-" * len(header)]
        for r in self.results:
            lines.append(
                f"{r.approach_name:<25}{r.ctr:>10.2%}{r.avg_dwell_ms:>18.1f}{r.return_rate:>15.2%}{r.diversity_score:>12.3f}"
            )
        two_tower = next((r for r in self.results if r.approach_name == "two_tower"), None)
        if two_tower:
            lines.append("")
            for r in self.results:
                if r.approach_name != "two_tower":
                    lines.append(f"two_tower CTR lift vs {r.approach_name}: {two_tower.ctr_lift_vs(r):+.1%}")
        return "\n".join(lines)
