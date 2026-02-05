"""Database connection and session management (async SQLAlchemy)."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..config.settings import get_settings
from ..models.database import Base

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


def session_factory() -> AsyncSession:
    """Create a new AsyncSession (caller must close)."""
    return AsyncSessionLocal()


async def init_db() -> None:
    """Initialize database (create tables) for MVP."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()


