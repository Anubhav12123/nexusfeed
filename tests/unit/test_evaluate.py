import numpy as np

from nexusfeed.models.evaluate import auc, catalog_coverage, intra_list_diversity, mrr, ndcg_at_k


def test_auc_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(labels, scores) == 1.0


def test_ndcg_at_k_perfect_ranking():
    relevances = [[1, 1, 0, 0]]
    assert ndcg_at_k(relevances, k=4) == 1.0


def test_ndcg_at_k_imperfect_ranking_is_lower():
    perfect = ndcg_at_k([[1, 1, 0, 0]], k=4)
    imperfect = ndcg_at_k([[0, 1, 0, 1]], k=4)
    assert imperfect < perfect


def test_mrr_first_relevant_item():
    relevances = [[0, 0, 1, 0], [1, 0, 0, 0]]
    result = mrr(relevances)
    # user 1: rank 3 -> 1/3; user 2: rank 1 -> 1
    assert abs(result - ((1 / 3 + 1) / 2)) < 1e-9


def test_catalog_coverage():
    assert catalog_coverage({"a", "b"}, total_catalog_size=10) == 0.2


def test_intra_list_diversity_identical_items_is_zero():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert intra_list_diversity(embeddings) < 0.01


def test_intra_list_diversity_orthogonal_items_is_high():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert intra_list_diversity(embeddings) > 0.9
