"""PostgreSQL + S3 historical feature access for training and cold-start lookups.

Hot features (recent, frequently accessed) live in Redis; cold features
(historical, used only for periodic retraining) live here. See blueprint
"Interview Trade-off to Know Cold" in Layer 2 for the reasoning.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexusfeed.config import Settings, get_settings
from nexusfeed.db.models import Interaction, Item, User


class OfflineFeatureStore:
    """Reads the long-tail historical interaction matrix used by:
    - the nightly/weekly full model retrain (PySpark/EMR in production, pandas locally)
    - cold-start content-based lookups for brand-new users
    """

    def __init__(self, session: AsyncSession, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()

    async def get_user_interaction_history(
        self, user_id: UUID, window_days: int = 30, limit: int = 500
    ) -> list[Interaction]:
        since = datetime.utcnow() - timedelta(days=window_days)
        stmt = (
            select(Interaction)
            .where(Interaction.user_id == user_id, Interaction.created_at >= since)
            .order_by(Interaction.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_item_historical_ctr(self, item_id: UUID) -> float:
        item = await self.session.get(Item, item_id)
        return item.historical_ctr if item else 0.0

    async def get_user_preferred_categories(self, user_id: UUID) -> list[str]:
        """Used for cold-start content-based fallback recommendations."""
        user = await self.session.get(User, user_id)
        if user is None:
            return []
        return (user.metadata_ or {}).get("onboarding_categories", [])

    async def get_training_window(self, since: datetime, until: datetime) -> list[Interaction]:
        """Pulls the interaction slice used to build a Parquet training snapshot.
        In production this reads S3 Parquet partitioned by date directly; locally
        we read from Postgres, which holds the same audit-trail data.
        """
        stmt = select(Interaction).where(
            Interaction.created_at >= since, Interaction.created_at < until
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
