"""Baseline 2: most-popular — ranks items purely by global interaction count.

No personalization at all; this is the baseline that a lazy "recommendation
system" would ship, and the one NexusFeed's diversity/CTR numbers need to
clearly beat to justify the added complexity.
"""
from __future__ import annotations

from collections import Counter


class PopularityBaseline:
    name = "most_popular"

    def __init__(self, interaction_counts: dict[str, int]) -> None:
        self.ranked_items = [item for item, _ in Counter(interaction_counts).most_common()]

    def recommend(self, user_id: str, n: int = 20) -> list[str]:
        return self.ranked_items[:n]
