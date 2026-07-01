"""Incremental mini-batch updates from the daily interaction stream.

Full retrain runs weekly on SageMaker (see training/sagemaker_job.py);
between full retrains, this module applies lightweight mini-batch SGD steps
on the current day's new interactions so the model doesn't drift too far
stale during the week.
"""
from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from nexusfeed.models.two_tower import TwoTowerModel

logger = logging.getLogger(__name__)


class OnlineLearner:
    def __init__(self, model: TwoTowerModel, learning_rate: float = 1e-4) -> None:
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def incremental_update(self, daily_loader: DataLoader) -> float:
        """One low-LR pass over today's interactions. Uses SGD (not Adam) —
        no momentum state to warm up, and a much smaller step size than the
        full weekly retrain, since the goal is drift correction, not
        re-learning from scratch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for user_batch, item_batch, negative_batch in daily_loader:
            self.optimizer.zero_grad()
            user_emb = self.model.encode_user(**user_batch)
            pos_item_emb = self.model.encode_item(**item_batch)

            batch_size, num_neg = negative_batch["item_ids"].shape
            flat_neg = {k: v.reshape(-1, *v.shape[2:]) for k, v in negative_batch.items()}
            neg_emb = self.model.encode_item(**flat_neg).reshape(batch_size, num_neg, -1)

            loss = self.model.sampled_softmax_loss(user_emb, pos_item_emb, neg_emb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info("online_learning_step_complete", extra={"avg_loss": avg_loss, "batches": n_batches})
        return avg_loss
