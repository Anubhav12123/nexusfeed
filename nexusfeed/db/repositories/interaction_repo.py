"""Async CRUD + aggregate queries for the (monthly-partitioned) interactions table."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nexusfeed.db.models import Interaction


class InteractionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record(
        self,
        user_id: UUID,
        item_id: UUID,
        event_type: str,
        dwell_ms: int | None = None,
        session_id: UUID | None = None,
    ) -> Interaction:
        interaction = Interaction(
            user_id=user_id, item_id=item_id, event_type=event_type, dwell_ms=dwell_ms, session_id=session_id
        )
        self.session.add(interaction)
        await self.session.flush()
        return interaction

    async def for_user(self, user_id: UUID, since: datetime, limit: int = 500) -> list[Interaction]:
        stmt = (
            select(Interaction)
            .where(Interaction.user_id == user_id, Interaction.created_at >= since)
            .order_by(Interaction.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_by_event_type(self, item_id: UUID) -> dict[str, int]:
        stmt = (
            select(Interaction.event_type, func.count())
            .where(Interaction.item_id == item_id)
            .group_by(Interaction.event_type)
        )
        result = await self.session.execute(stmt)
        return dict(result.all())

    async def click_through_rate(self, item_id: UUID) -> float:
        counts = await self.count_by_event_type(item_id)
        impressions = counts.get("view_full", 0) + counts.get("view_scroll_past", 0) + counts.get("click", 0)
        clicks = counts.get("click", 0)
        return clicks / impressions if impressions else 0.0
