"""Baseline 1: random recommendation — the floor every other approach must beat."""
from __future__ import annotations

import random


class RandomBaseline:
    name = "random"

    def __init__(self, item_ids: list[str]) -> None:
        self.item_ids = item_ids

    def recommend(self, user_id: str, n: int = 20) -> list[str]:
        return random.sample(self.item_ids, min(n, len(self.item_ids)))
