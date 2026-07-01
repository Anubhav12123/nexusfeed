"""Health check with component status — p99 < 2ms target."""
from __future__ import annotations

from fastapi import APIRouter, Request

from nexusfeed import __version__
from nexusfeed.types import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
async def health(request: Request) -> HealthStatus:
    components = {"api": "ok"}

    redis = getattr(request.app.state, "redis", None)
    if redis is not None:
        try:
            await redis.ping()
            components["redis"] = "ok"
        except Exception:  # noqa: BLE001
            components["redis"] = "down"

    faiss_index = getattr(request.app.state, "faiss_index", None)
    components["faiss_index"] = "ready" if (faiss_index and faiss_index.is_ready) else "not_ready"

    db = getattr(request.app.state, "db", None)
    components["postgres"] = "ok" if db is not None else "unconfigured"

    overall = "ok" if all(v in ("ok", "ready") for v in components.values()) else "degraded"
    return HealthStatus(status=overall, components=components, version=__version__)
