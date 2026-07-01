"""GET /similar/{item_id} — returns similar items via ANN retrieval. p99 < 20ms."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from nexusfeed.db.connection import get_db
from nexusfeed.db.repositories.item_repo import ItemRepository
from nexusfeed.types import FeedItem, SimilarItemsResponse

router = APIRouter(tags=["similar"])


@router.get("/similar/{item_id}", response_model=SimilarItemsResponse)
async def get_similar_items(
    item_id: UUID,
    request: Request,
    n: int = 10,
    db: AsyncSession = Depends(get_db),
) -> SimilarItemsResponse:
    index = request.app.state.faiss_index
    repo = ItemRepository(db)
    item = await repo.get(item_id)

    if item is None or item.embedding is None:
        return SimilarItemsResponse(item_id=item_id, similar_items=[])

    if index.is_ready:
        results = index.search(__import__("numpy").array(item.embedding), k=n + 1)
        results = [(iid, score) for iid, score in results if iid != str(item_id)][:n]
    else:
        similar = await repo.most_similar(item.embedding, k=n, exclude=item_id)
        results = [(str(i.id), 1.0) for i in similar]

    items = [FeedItem(item_id=iid, score=score, rank=idx + 1) for idx, (iid, score) in enumerate(results)]
    return SimilarItemsResponse(item_id=item_id, similar_items=items)
