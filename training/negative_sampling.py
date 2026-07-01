"""In-batch negative sampling for efficient two-tower training.

Ratio 1:4 positive-to-negative per the blueprint's training pipeline spec.
In-batch negatives (rather than globally sampled ones) avoid a second data
fetch per step and give harder negatives as batch size grows.
"""
from __future__ import annotations

import random

import numpy as np

NEGATIVE_RATIO = 4


def sample_in_batch_negatives(
    positive_item_indices: np.ndarray, num_items: int, ratio: int = NEGATIVE_RATIO
) -> np.ndarray:
    """Returns an (batch_size, ratio) array of negative item indices sampled
    uniformly at random, excluding each row's own positive index.
    """
    batch_size = len(positive_item_indices)
    negatives = np.zeros((batch_size, ratio), dtype=np.int64)
    for i, positive in enumerate(positive_item_indices):
        chosen: list[int] = []
        while len(chosen) < ratio:
            candidate = random.randint(0, num_items - 1)
            if candidate != positive:
                chosen.append(candidate)
        negatives[i] = chosen
    return negatives


def sample_shown_not_clicked_negatives(
    impressions: list[tuple[int, list[int]]], clicked_item: int, ratio: int = NEGATIVE_RATIO
) -> list[int]:
    """Prefers negatives from items that were actually shown to the user but
    not clicked (harder negatives than uniform random) — falls back to
    uniform random sampling if fewer than `ratio` shown-not-clicked items
    exist for this impression.
    """
    shown_not_clicked = [item for _, shown_items in impressions for item in shown_items if item != clicked_item]
    if len(shown_not_clicked) >= ratio:
        return random.sample(shown_not_clicked, ratio)
    return shown_not_clicked
