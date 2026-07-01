"""Async CRUD repository for users."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexusfeed.db.models import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, metadata: dict | None = None) -> User:
        user = User(metadata_=metadata or {})
        self.session.add(user)
        await self.session.flush()
        return user

    async def get(self, user_id: UUID) -> User | None:
        return await self.session.get(User, user_id)

    async def list_recent(self, limit: int = 100) -> list[User]:
        stmt = select(User).order_by(User.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_metadata(self, user_id: UUID, metadata: dict) -> User | None:
        user = await self.get(user_id)
        if user is None:
            return None
        user.metadata_ = {**user.metadata_, **metadata}
        await self.session.flush()
        return user
