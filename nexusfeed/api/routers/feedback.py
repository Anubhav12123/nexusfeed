"""POST /feedback — explicit feedback (like, dislike, report). p99 < 10ms."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from nexusfeed.db.connection import get_db
from nexusfeed.db.repositories.interaction_repo import InteractionRepository
from nexusfeed.features.online_store import OnlineFeatureStore
from nexusfeed.types import EventIngestResponse, FeedbackRequest

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=EventIngestResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> EventIngestResponse:
    repo = InteractionRepository(db)
    interaction = await repo.record(
        user_id=feedback.user_id,
        item_id=feedback.item_id,
        event_type=feedback.feedback_type.value,
    )
    await db.commit()

    # Immediate re-ranking penalty for report/hide, per blueprint Layer 1 event table.
    if feedback.feedback_type.value in ("report", "hide"):
        online_store = OnlineFeatureStore(request.app.state.redis)
        await online_store.set_item_score(feedback.item_id, -1.0)

    return EventIngestResponse(event_id=interaction.id, status="accepted")
