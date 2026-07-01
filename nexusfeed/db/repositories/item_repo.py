"""Async CRUD repository for items, including pgvector similarity queries."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from nexusfeed.db.models import Item


class ItemRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, category: str, content_text: str | None = None, embedding: list[float] | None = None
    ) -> Item:
        item = Item(category=category, content_text=content_text, embedding=embedding)
        self.session.add(item)
        await self.session.flush()
        return item

    async def get(self, item_id: UUID) -> Item | None:
        return await self.session.get(Item, item_id)

    async def get_many(self, item_ids: list[UUID]) -> list[Item]:
        stmt = select(Item).where(Item.id.in_(item_ids))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def most_similar(
        self, embedding: list[float], k: int = 10, exclude: UUID | None = None
    ) -> list[Item]:
        """Fallback pgvector cosine-distance query — used when the FAISS
        index is cold (e.g. right after a fresh deploy before the nightly
        index build has run once).
        """
        stmt = select(Item).order_by(Item.embedding.cosine_distance(embedding)).limit(k)
        if exclude is not None:
            stmt = stmt.where(Item.id != exclude)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def increment_impressions(self, item_id: UUID, by: int = 1) -> None:
        stmt = update(Item).where(Item.id == item_id).values(impression_count=Item.impression_count + by)
        await self.session.execute(stmt)

    async def update_ctr(self, item_id: UUID, ctr: float) -> None:
        stmt = update(Item).where(Item.id == item_id).values(historical_ctr=ctr)
        await self.session.execute(stmt)

    async def by_category(self, category: str, limit: int = 100) -> list[Item]:
        stmt = select(Item).where(Item.category == category).order_by(Item.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
