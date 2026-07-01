"""Epsilon-greedy multi-armed bandit for item-level exploration.

10% of feed slots serve randomly sampled items to collect signal on new
content that the two-tower model hasn't seen enough interactions for yet.
This is distinct from the cold-start forced-exploration heuristic in
item_features.py — the bandit operates at the *feed slot* level continuously,
not just for brand-new items.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ArmStats:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0


class EpsilonGreedyBandit:
    def __init__(self, epsilon: float = 0.10) -> None:
        self.epsilon = epsilon
        self.arms: dict[str, ArmStats] = {}

    def register_arm(self, item_id: str) -> None:
        self.arms.setdefault(item_id, ArmStats())

    def record_reward(self, item_id: str, reward: float) -> None:
        stats = self.arms.setdefault(item_id, ArmStats())
        stats.pulls += 1
        stats.total_reward += reward

    def select_exploration_slot(self, candidate_pool: list[str]) -> str | None:
        """With probability epsilon, explore uniformly at random from the
        candidate pool; otherwise exploit the best-known arm among them.
        """
        if not candidate_pool:
            return None
        if random.random() < self.epsilon:
            return random.choice(candidate_pool)

        known = [
            (item_id, self.arms[item_id].mean_reward) for item_id in candidate_pool if item_id in self.arms
        ]
        if not known:
            return random.choice(candidate_pool)
        return max(known, key=lambda x: x[1])[0]

    def inject_exploration_slots(
        self, feed_item_ids: list[str], candidate_pool: list[str], slot_fraction: float | None = None
    ) -> list[str]:
        """Replaces a fraction of the ranked feed with bandit-selected
        exploration items, preserving the rest of the ranking order.
        """
        fraction = slot_fraction if slot_fraction is not None else self.epsilon
        n_slots = max(1, int(len(feed_item_ids) * fraction)) if feed_item_ids else 0
        result = list(feed_item_ids)
        available = [c for c in candidate_pool if c not in result]

        for _ in range(min(n_slots, len(available))):
            slot = self.select_exploration_slot(available)
            if slot is None:
                break
            # Replace the lowest-ranked (last) slot rather than disturbing top ranks.
            if result:
                result[-1] = slot
            available.remove(slot)
        return result
