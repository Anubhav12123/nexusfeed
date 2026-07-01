"""FastAPI app factory with lifespan event handlers.

Wires every layer of the blueprint's seven-layer architecture into app.state
so routers can reach them via `request.app.state.*` without a heavyweight DI
container. Kept intentionally simple: a portfolio project should be readable
top to bottom in one sitting.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from redis.asyncio import Redis

from nexusfeed.api.middleware.auth import ApiKeyAuthMiddleware
from nexusfeed.api.middleware.logging import AccessLogMiddleware
from nexusfeed.api.middleware.rate_limit import RateLimitMiddleware
from nexusfeed.api.middleware.request_id import RequestIdMiddleware
from nexusfeed.api.routers import admin, events, experiments, explain, feed, feedback, health, similar
from nexusfeed.config import get_settings
from nexusfeed.db.connection import get_database
from nexusfeed.ingestion.kafka_producer import KafkaEventProducer
from nexusfeed.observability.logging import configure_logging
from nexusfeed.retrieval.candidate_generator import CandidateGenerator
from nexusfeed.retrieval.faiss_index import FaissIndex

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, json_format=(settings.log_format == "json"))

    app.state.settings = settings
    app.state.embedding_dim = settings.embedding_dim
    app.state.model_version = settings.fallback_model_version
    try:
        from nexusfeed.models.model_registry import ModelRegistry

        production_version = ModelRegistry(settings).get_production_version(settings.ranking_model_name)
        if production_version:
            app.state.model_version = f"{settings.ranking_model_name}-v{production_version}"
    except Exception:  # noqa: BLE001 - MLflow registry not reachable yet on a fresh install
        logger.warning("model_registry_unavailable_using_fallback_version")

    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)

    app.state.db = get_database()

    app.state.kafka_producer = KafkaEventProducer(settings)
    try:
        await app.state.kafka_producer.start()
    except Exception as exc:  # noqa: BLE001 - allow API to boot without Kafka in constrained dev envs
        logger.warning("kafka_unavailable_at_startup", extra={"error": str(exc)})

    app.state.faiss_index = FaissIndex(dim=settings.embedding_dim)
    try:
        app.state.faiss_index.load(settings.ann_index_path)
        logger.info("faiss_index_loaded", extra={"size": app.state.faiss_index.size})
    except Exception:  # noqa: BLE001 - fine at first boot before any index has been built
        logger.warning("faiss_index_not_found_serving_will_use_cold_start_fallback")

    app.state.candidate_generator = CandidateGenerator(app.state.faiss_index)

    app.state.ranker = None
    app.state.shap_explainer = None
    try:
        from nexusfeed.models.ranking_model import RankingModel
        from nexusfeed.ranking.ranker import Ranker

        ranking_model = RankingModel.load(settings.ranking_model_path)
        app.state.ranker = Ranker(ranking_model)

        from nexusfeed.explainability.shap_explainer import ShapExplainer

        app.state.shap_explainer = ShapExplainer(ranking_model.booster)
        logger.info("ranking_model_loaded", extra={"path": settings.ranking_model_path})
    except Exception:  # noqa: BLE001 - fine before scripts/seed_faiss_index.py has run once
        logger.warning("ranking_model_not_found_using_ann_score_fallback")

    yield

    await app.state.kafka_producer.stop()
    await app.state.redis.aclose()
    await app.state.db.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="NexusFeed",
        description="Real-Time AI Recommendation and Personalization Engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(ApiKeyAuthMiddleware)
    app.add_middleware(RequestIdMiddleware)

    app.include_router(health.router)
    app.include_router(feed.router)
    app.include_router(events.router)
    app.include_router(similar.router)
    app.include_router(feedback.router)
    app.include_router(experiments.router)
    app.include_router(explain.router)
    app.include_router(admin.router)

    app.mount("/metrics", make_asgi_app())

    return app


app = create_app()
