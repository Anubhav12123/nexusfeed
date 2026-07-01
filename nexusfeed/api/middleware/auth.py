"""API key authentication with scope-based permissions."""
from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from nexusfeed.config import get_settings

PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/metrics"}


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        # app.mount("/metrics", ...) makes Starlette 307-redirect a bare
        # "/metrics" request to "/metrics/" — Prometheus follows the
        # redirect and re-requests the trailing-slash path, which wouldn't
        # match PUBLIC_PATHS by exact string and would 401 every scrape.
        if request.url.path.rstrip("/") in PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get(settings.api_key_header)
        if api_key is None or api_key not in settings.api_keys:
            return JSONResponse(status_code=401, content={"detail": "invalid or missing API key"})

        request.state.api_key = api_key
        return await call_next(request)
