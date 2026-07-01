"""Offline evaluation: AUC, NDCG@10, MRR, coverage, diversity."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def ndcg_at_k(relevances: list[list[int]], k: int = 10) -> float:
    """relevances: per-user list of binary/graded relevance labels, already
    sorted in the order the model ranked them (index 0 = top rank).
    """
    def dcg(rels: list[int]) -> float:
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rels[:k]))

    scores = []
    for rels in relevances:
        ideal = sorted(rels, reverse=True)
        ideal_dcg = dcg(ideal)
        scores.append(dcg(rels) / ideal_dcg if ideal_dcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def mrr(relevances: list[list[int]]) -> float:
    """Mean reciprocal rank of the first relevant item per user."""
    reciprocal_ranks = []
    for rels in relevances:
        rank = next((i + 1 for i, r in enumerate(rels) if r > 0), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def catalog_coverage(recommended_item_ids: set[str], total_catalog_size: int) -> float:
    return len(recommended_item_ids) / total_catalog_size if total_catalog_size else 0.0


def intra_list_diversity(item_embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance within a recommended list — higher means
    a more diverse feed. Used to validate MMR re-ranking isn't a no-op.
    """
    if len(item_embeddings) < 2:
        return 0.0
    normed = item_embeddings / (np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-9)
    sim_matrix = normed @ normed.T
    n = len(item_embeddings)
    off_diagonal_sum = sim_matrix.sum() - np.trace(sim_matrix)
    mean_sim = off_diagonal_sum / (n * (n - 1))
    return float(1 - mean_sim)


class EvaluationReport:
    def __init__(
        self, auc_score: float, ndcg10: float, mrr_score: float, coverage: float, diversity: float
    ) -> None:
        self.auc = auc_score
        self.ndcg10 = ndcg10
        self.mrr = mrr_score
        self.coverage = coverage
        self.diversity = diversity

    def as_dict(self) -> dict[str, float]:
        return {
            "auc": self.auc,
            "ndcg@10": self.ndcg10,
            "mrr": self.mrr,
            "coverage": self.coverage,
            "diversity": self.diversity,
        }
