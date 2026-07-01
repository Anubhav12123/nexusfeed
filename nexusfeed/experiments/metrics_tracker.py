"""Per-experiment CTR, dwell time, return rate tracking."""
from __future__ import annotations

from uuid import UUID

from nexusfeed.db.repositories.experiment_repo import ExperimentRepository
from nexusfeed.types import EventType, ExperimentMetrics


class MetricsTracker:
    def __init__(self, repo: ExperimentRepository) -> None:
        self.repo = repo

    async def record_impression(self, experiment_id: UUID, variant: str) -> None:
        await self.repo.record_impression(experiment_id, variant)

    async def record_event(
        self, experiment_id: UUID, variant: str, event_type: EventType, dwell_ms: int | None = None
    ) -> None:
        if event_type == EventType.CLICK:
            await self.repo.record_click(experiment_id, variant)
        elif event_type == EventType.SHARE or event_type == EventType.SAVE:
            await self.repo.record_share(experiment_id, variant)
        elif event_type == EventType.DWELL and dwell_ms:
            await self.repo.record_dwell(experiment_id, variant, dwell_ms)

    async def get_metrics(self, experiment_id: UUID) -> dict[str, ExperimentMetrics]:
        results = await self.repo.results_for(experiment_id)
        return {
            row.variant: ExperimentMetrics(
                experiment_name=str(experiment_id),
                variant=row.variant,
                impressions=row.impressions,
                clicks=row.clicks,
                shares=row.shares,
                dwell_ms_total=row.dwell_ms_total,
                returns=row.returns,
            )
            for row in results
        }
