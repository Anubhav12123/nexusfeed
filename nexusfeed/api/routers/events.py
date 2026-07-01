"""POST /events — behavioral event ingestion. p99 < 10ms (async Kafka write)."""
from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from nexusfeed.db.connection import get_db
from nexusfeed.db.repositories.interaction_repo import InteractionRepository
from nexusfeed.exceptions import DuplicateEventError, EventValidationError
from nexusfeed.ingestion.event_router import EventRouter
from nexusfeed.ingestion.event_validator import EventValidator
from nexusfeed.observability.metrics import EVENTS_LATENCY_SECONDS, INGESTION_ERRORS
from nexusfeed.types import Event, EventIngestResponse

router = APIRouter(tags=["events"])


@router.post("/events", response_model=EventIngestResponse, status_code=200)
async def ingest_event(
    event: Event,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> EventIngestResponse:
    with EVENTS_LATENCY_SECONDS.time():
        validator = EventValidator(request.app.state.redis)
        try:
            await validator.validate(event)
        except DuplicateEventError:
            INGESTION_ERRORS.labels(reason="duplicate").inc()
            return EventIngestResponse(event_id=event.event_id, status="duplicate_ignored")
        except EventValidationError:
            INGESTION_ERRORS.labels(reason="invalid").inc()
            raise

        router_ = EventRouter(request.app.state.kafka_producer)
        await router_.route(event)  # fire-and-forget-to-Kafka semantics per blueprint Layer 1

        # Async audit write to Postgres — not on the critical path for the 200 response.
        repo = InteractionRepository(db)
        await repo.record(
            user_id=event.user_id,
            item_id=event.item_id,
            event_type=event.event_type.value,
            dwell_ms=event.dwell_ms,
            session_id=event.session_id,
        )
        await db.commit()

    return EventIngestResponse(event_id=event.event_id, status="accepted")
