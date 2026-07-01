"""Async SQLAlchemy engine with connection pooling."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nexusfeed.config import Settings, get_settings


class Database:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine = create_async_engine(
            self.settings.database_url,
            pool_size=self.settings.database_pool_size,
            max_overflow=self.settings.database_max_overflow,
            pool_pre_ping=True,
        )
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def dispose(self) -> None:
        await self.engine.dispose()


_db_singleton: Database | None = None


def get_database() -> Database:
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = Database()
    return _db_singleton


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    db = get_database()
    async with db.session_factory() as session:
        yield session
