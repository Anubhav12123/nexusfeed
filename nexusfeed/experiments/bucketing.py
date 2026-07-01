"""Deterministic user bucketing via hash — consistent assignment across sessions.

A user always lands in the same bucket (0-99) for a given experiment name,
because the hash input is `(experiment_name, user_id)`. This is what makes
"control: 0-49, treatment: 50-99" configuration meaningful and stable.
"""
from __future__ import annotations

import hashlib
from uuid import UUID

HOLDBACK_FRACTION_DEFAULT = 0.05


def bucket_for(user_id: UUID, experiment_name: str, num_buckets: int = 100) -> int:
    digest = hashlib.sha256(f"{experiment_name}:{user_id}".encode()).hexdigest()
    return int(digest, 16) % num_buckets


def is_in_holdback(user_id: UUID, holdback_fraction: float = HOLDBACK_FRACTION_DEFAULT) -> bool:
    """5% of users always receive the baseline non-personalized feed,
    independent of any specific experiment — the long-term comparison
    baseline described in blueprint Layer 7.
    """
    digest = hashlib.sha256(f"__global_holdback__:{user_id}".encode()).hexdigest()
    bucket = int(digest, 16) % 10000
    return bucket < int(holdback_fraction * 10000)


def assign_variant(
    bucket: int, control_range: tuple[int, int], treatment_range: tuple[int, int]
) -> str:
    if control_range[0] <= bucket < control_range[1]:
        return "control"
    if treatment_range[0] <= bucket < treatment_range[1]:
        return "treatment"
    return "unassigned"
