"""Redis sliding window rate limiter, returns 429 with Retry-After."""
from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from nexusfeed.config import get_settings
from nexusfeed.observability.metrics import RATE_LIMIT_REJECTIONS


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if not request.url.path.startswith("/events"):
            return await call_next(request)

        redis = getattr(request.app.state, "redis", None)
        api_key = getattr(request.state, "api_key", "anonymous")
        if redis is None:
            return await call_next(request)

        from nexusfeed.features.online_store import OnlineFeatureStore

        store = OnlineFeatureStore(redis, settings)
        allowed, count = await store.rate_limit_hit(
            api_key, settings.rate_limit_window_seconds, settings.rate_limit_events_per_second
        )
        if not allowed:
            RATE_LIMIT_REJECTIONS.inc()
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded"},
                headers={"Retry-After": str(settings.rate_limit_window_seconds)},
            )
        return await call_next(request)
