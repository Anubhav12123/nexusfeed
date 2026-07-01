"""Nightly batch rebuild of the FAISS index from updated item embeddings.

Also implements the multi-index-per-category strategy mentioned in the
blueprint (Layer 4): separate FAISS indices per category enable category-
aware retrieval and much faster incremental rebuilds than one giant index.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexusfeed.db.models import Item
from nexusfeed.retrieval.faiss_index import FaissIndex

logger = logging.getLogger(__name__)


class IndexBuilder:
    def __init__(self, session: AsyncSession, embedding_dim: int = 128) -> None:
        self.session = session
        self.embedding_dim = embedding_dim

    async def _load_embedded_items(self) -> list[Item]:
        stmt = select(Item).where(Item.embedding.is_not(None))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def build_global_index(self) -> FaissIndex:
        items = await self._load_embedded_items()
        index = FaissIndex(dim=self.embedding_dim)
        if not items:
            logger.warning("index_builder_no_embedded_items")
            return index
        embeddings = np.array([item.embedding for item in items], dtype=np.float32)
        ids = [str(item.id) for item in items]
        index.build(ids, embeddings)
        return index

    async def build_category_indices(self) -> dict[str, FaissIndex]:
        items = await self._load_embedded_items()
        by_category: dict[str, list[Item]] = defaultdict(list)
        for item in items:
            by_category[item.category].append(item)

        indices: dict[str, FaissIndex] = {}
        for category, category_items in by_category.items():
            index = FaissIndex(dim=self.embedding_dim)
            embeddings = np.array([i.embedding for i in category_items], dtype=np.float32)
            ids = [str(i.id) for i in category_items]
            index.build(ids, embeddings)
            indices[category] = index
            logger.info("category_index_built", extra={"category": category, "n_items": len(category_items)})
        return indices
