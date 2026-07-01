import pytest

from nexusfeed.retrieval.candidate_generator import CandidateGenerator
from nexusfeed.retrieval.faiss_index import FaissIndex


def _built_index(random_embeddings, n=20, dim=16):
    embeddings = random_embeddings(n, dim=dim)
    ids = [f"item_{i}" for i in range(n)]
    index = FaissIndex(dim=dim)
    index.build(ids, embeddings)
    return index, embeddings


@pytest.mark.asyncio
async def test_retrieve_returns_ann_results(random_embeddings):
    index, embeddings = _built_index(random_embeddings)
    generator = CandidateGenerator(index)
    results = await generator.retrieve(embeddings[0].tolist(), k=5)
    assert len(results) == 5
    assert "item_0" in {item_id for item_id, _ in results}


@pytest.mark.asyncio
async def test_retrieve_injects_forced_exploration_items_not_in_ann_results(random_embeddings):
    index, embeddings = _built_index(random_embeddings)
    generator = CandidateGenerator(index)
    results = await generator.retrieve(
        embeddings[0].tolist(), k=3, forced_exploration_item_ids=["brand_new_item"]
    )
    result_ids = {item_id for item_id, _ in results}
    assert "brand_new_item" in result_ids


@pytest.mark.asyncio
async def test_retrieve_injects_trending_items(random_embeddings):
    index, embeddings = _built_index(random_embeddings)
    generator = CandidateGenerator(index)
    results = await generator.retrieve(embeddings[0].tolist(), k=3, trending_item_ids=["trending_item"])
    result_ids = {item_id for item_id, _ in results}
    assert "trending_item" in result_ids


@pytest.mark.asyncio
async def test_retrieve_does_not_duplicate_items_already_in_ann_results(random_embeddings):
    index, embeddings = _built_index(random_embeddings, n=5)
    generator = CandidateGenerator(index)
    results = await generator.retrieve(embeddings[0].tolist(), k=5, trending_item_ids=["item_1"])
    result_ids = [item_id for item_id, _ in results]
    assert result_ids.count("item_1") == 1
