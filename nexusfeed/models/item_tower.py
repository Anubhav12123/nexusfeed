"""Item encoder — ID embedding + category + BERT content embedding + freshness/CTR."""
from __future__ import annotations

import torch
from torch import nn


class ItemTower(nn.Module):
    """`content_embedding` is expected to be a precomputed 768-dim BERT
    (or similar) embedding for text items, produced once at item-creation
    time and cached — never computed inline in the forward pass, since that
    would blow the serving latency budget by orders of magnitude.
    """

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        embedding_dim: int = 128,
        content_embedding_dim: int = 768,
    ) -> None:
        super().__init__()
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, 16)
        self.content_projection = nn.Linear(content_embedding_dim, embedding_dim)

        combined_dim = embedding_dim + 16 + embedding_dim + 2  # id + category + content + (freshness, ctr)
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor,
        content_embedding: torch.Tensor,
        freshness_score: torch.Tensor,
        historical_ctr: torch.Tensor,
    ) -> torch.Tensor:
        item_emb = self.item_id_embedding(item_ids)
        category_emb = self.category_embedding(category_ids)
        content_emb = torch.tanh(self.content_projection(content_embedding))
        scalar_feats = torch.stack([freshness_score, historical_ctr], dim=-1)

        combined = torch.cat([item_emb, category_emb, content_emb, scalar_feats], dim=-1)
        out = self.projection(combined)
        return nn.functional.normalize(out, p=2, dim=-1)
