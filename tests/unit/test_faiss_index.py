import numpy as np
import pytest

from nexusfeed.exceptions import IndexNotReadyError
from nexusfeed.retrieval.faiss_index import FaissIndex


def test_search_before_build_raises():
    index = FaissIndex(dim=16)
    with pytest.raises(IndexNotReadyError):
        index.search(np.zeros(16))


def test_build_and_search_returns_nearest_neighbor(random_embeddings):
    embeddings = random_embeddings(100, dim=16)
    ids = [f"item_{i}" for i in range(100)]

    index = FaissIndex(dim=16)
    index.build(ids, embeddings)

    query = embeddings[5]
    results = index.search(query, k=5)
    result_ids = [item_id for item_id, _ in results]
    assert "item_5" in result_ids
    assert result_ids[0] == "item_5"  # itself should be the closest match


def test_hot_swap_replaces_results(random_embeddings):
    index = FaissIndex(dim=16)
    embeddings_v1 = random_embeddings(50, dim=16)
    index.build([f"v1_{i}" for i in range(50)], embeddings_v1)
    assert index.size == 50

    embeddings_v2 = random_embeddings(80, dim=16)
    index.build([f"v2_{i}" for i in range(80)], embeddings_v2)
    assert index.size == 80

    results = index.search(embeddings_v2[0], k=1)
    assert results[0][0].startswith("v2_")
