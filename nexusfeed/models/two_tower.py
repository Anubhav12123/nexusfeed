"""Two-tower recommendation model — the industry-standard architecture used
at YouTube, Pinterest, TikTok, and Spotify. Dot product of the two 128-dim
embeddings scores relevance; the towers can run independently, which is what
makes fast ANN retrieval possible (see blueprint Layer 3 + the interview
answer "why two-tower over a single concat model").
"""
from __future__ import annotations

import torch
from torch import nn

from nexusfeed.models.item_tower import ItemTower
from nexusfeed.models.user_tower import UserTower


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        embedding_dim: int = 128,
        content_embedding_dim: int = 768,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)
        self.item_tower = ItemTower(
            num_items=num_items, num_categories=num_categories,
            embedding_dim=embedding_dim, content_embedding_dim=content_embedding_dim,
        )

    def encode_user(self, *args, **kwargs) -> torch.Tensor:
        return self.user_tower(*args, **kwargs)

    def encode_item(self, *args, **kwargs) -> torch.Tensor:
        return self.item_tower(*args, **kwargs)

    def forward(self, user_batch: dict, item_batch: dict) -> torch.Tensor:
        """Returns the interaction-layer relevance score: the dot product of
        the two embeddings passed through a sigmoid (binary cross-entropy on
        click vs no-click, per the blueprint's Interaction Layer spec).
        """
        user_emb = self.encode_user(**user_batch)
        item_emb = self.encode_item(**item_batch)
        logits = (user_emb * item_emb).sum(dim=-1)
        return logits

    def sampled_softmax_loss(
        self, user_emb: torch.Tensor, positive_item_emb: torch.Tensor, in_batch_negatives: torch.Tensor
    ) -> torch.Tensor:
        """Sampled softmax over in-batch negatives — more efficient than
        pointwise BCE for large item catalogs (blueprint Training Pipeline).
        `in_batch_negatives` is (batch, num_negatives, dim).
        """
        pos_logits = (user_emb * positive_item_emb).sum(dim=-1, keepdim=True)
        neg_logits = torch.einsum("bd,bnd->bn", user_emb, in_batch_negatives)
        all_logits = torch.cat([pos_logits, neg_logits], dim=-1)
        labels = torch.zeros(all_logits.size(0), dtype=torch.long, device=all_logits.device)
        return nn.functional.cross_entropy(all_logits, labels)
