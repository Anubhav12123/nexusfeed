"""CRUD for experiment configs, with Redis caching of per-user assignments."""
from __future__ import annotations

from uuid import UUID

from redis.asyncio import Redis

from nexusfeed.db.models import Experiment
from nexusfeed.db.repositories.experiment_repo import ExperimentRepository
from nexusfeed.exceptions import ExperimentNotFoundError
from nexusfeed.experiments.bucketing import assign_variant, bucket_for, is_in_holdback
from nexusfeed.observability.metrics import EXPERIMENT_ASSIGNMENTS
from nexusfeed.types import ExperimentAssignment


class ExperimentManager:
    def __init__(self, repo: ExperimentRepository, redis: Redis) -> None:
        self.repo = repo
        self.redis = redis

    async def create_experiment(
        self, name: str, control_range: tuple[int, int], treatment_range: tuple[int, int], config: dict
    ) -> Experiment:
        return await self.repo.create(name, control_range, treatment_range, config)

    async def get_assignment(self, user_id: UUID, experiment_name: str) -> ExperimentAssignment:
        if is_in_holdback(user_id):
            EXPERIMENT_ASSIGNMENTS.labels(experiment=experiment_name, variant="holdback").inc()
            return ExperimentAssignment(
                user_id=user_id, experiment_name=experiment_name, bucket=-1, variant="holdback"
            )

        cache_key = f"experiment:{experiment_name}:user:{user_id}"
        cached = await self.redis.get(cache_key)
        if cached is not None:
            bucket, variant = cached.split(":")
            return ExperimentAssignment(
                user_id=user_id, experiment_name=experiment_name, bucket=int(bucket), variant=variant
            )

        experiment = await self.repo.get_by_name(experiment_name)
        if experiment is None:
            raise ExperimentNotFoundError(f"experiment '{experiment_name}' not found")

        bucket = bucket_for(user_id, experiment_name)
        variant = assign_variant(
            bucket,
            (experiment.control_low, experiment.control_high),
            (experiment.treatment_low, experiment.treatment_high),
        )

        await self.redis.set(cache_key, f"{bucket}:{variant}", ex=604800)
        EXPERIMENT_ASSIGNMENTS.labels(experiment=experiment_name, variant=variant).inc()
        return ExperimentAssignment(
            user_id=user_id, experiment_name=experiment_name, bucket=bucket, variant=variant
        )

    async def list_active(self) -> list[Experiment]:
        return await self.repo.list_active()

    async def stop_experiment(self, experiment_name: str) -> None:
        from datetime import datetime

        experiment = await self.repo.get_by_name(experiment_name)
        if experiment is None:
            raise ExperimentNotFoundError(f"experiment '{experiment_name}' not found")
        experiment.status = "stopped"
        experiment.ended_at = datetime.utcnow()
        await self.repo.session.flush()
