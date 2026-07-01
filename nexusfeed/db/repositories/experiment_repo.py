"""Async CRUD for experiment configs and per-variant result rollups."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexusfeed.db.models import Experiment, ExperimentResult


class ExperimentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        name: str,
        control_range: tuple[int, int],
        treatment_range: tuple[int, int],
        config: dict,
    ) -> Experiment:
        experiment = Experiment(
            name=name,
            control_low=control_range[0],
            control_high=control_range[1],
            treatment_low=treatment_range[0],
            treatment_high=treatment_range[1],
            config=config,
        )
        self.session.add(experiment)
        await self.session.flush()
        for variant in ("control", "treatment", "holdback"):
            self.session.add(ExperimentResult(experiment_id=experiment.id, variant=variant))
        await self.session.flush()
        return experiment

    async def get_by_name(self, name: str) -> Experiment | None:
        stmt = select(Experiment).where(Experiment.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_active(self) -> list[Experiment]:
        stmt = select(Experiment).where(Experiment.status == "active")
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def record_impression(self, experiment_id: UUID, variant: str) -> None:
        await self._bump(experiment_id, variant, impressions=1)

    async def record_click(self, experiment_id: UUID, variant: str) -> None:
        await self._bump(experiment_id, variant, clicks=1)

    async def record_share(self, experiment_id: UUID, variant: str) -> None:
        await self._bump(experiment_id, variant, shares=1)

    async def record_dwell(self, experiment_id: UUID, variant: str, dwell_ms: int) -> None:
        await self._bump(experiment_id, variant, dwell_ms_total=dwell_ms)

    async def _bump(self, experiment_id: UUID, variant: str, **deltas: int) -> None:
        stmt = select(ExperimentResult).where(
            ExperimentResult.experiment_id == experiment_id, ExperimentResult.variant == variant
        )
        result = await self.session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return
        for field, delta in deltas.items():
            setattr(row, field, getattr(row, field) + delta)
        await self.session.flush()

    async def results_for(self, experiment_id: UUID) -> list[ExperimentResult]:
        stmt = select(ExperimentResult).where(ExperimentResult.experiment_id == experiment_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
