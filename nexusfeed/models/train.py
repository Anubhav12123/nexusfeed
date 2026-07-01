"""Full training loop with MLflow tracking and early stopping.

Positive examples are clicked items; negatives are sampled shown-but-not-
clicked at a 1:4 ratio (blueprint Training Pipeline spec). Loss is sampled
softmax over in-batch negatives, which scales far better than pointwise BCE
once the item catalog gets large.
"""
from __future__ import annotations

import logging

import mlflow
import torch
from torch.utils.data import DataLoader

from nexusfeed.models.two_tower import TwoTowerModel

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: TwoTowerModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            user_batch, item_batch, negative_batch = batch
            user_batch = {k: v.to(self.device) for k, v in user_batch.items()}
            item_batch = {k: v.to(self.device) for k, v in item_batch.items()}
            negative_batch = {k: v.to(self.device) for k, v in negative_batch.items()}

            self.optimizer.zero_grad()
            user_emb = self.model.encode_user(**user_batch)
            pos_item_emb = self.model.encode_item(**item_batch)

            batch_size, num_neg = negative_batch["item_ids"].shape
            flat_neg = {k: v.reshape(-1, *v.shape[2:]) for k, v in negative_batch.items()}
            neg_emb_flat = self.model.encode_item(**flat_neg)
            neg_emb = neg_emb_flat.reshape(batch_size, num_neg, -1)

            loss = self.model.sampled_softmax_loss(user_emb, pos_item_emb, neg_emb)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int = 20,
        patience: int = 3,
        experiment_name: str = "nexusfeed-two-tower",
    ) -> dict[str, float]:
        mlflow.set_experiment(experiment_name)
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history: dict[str, float] = {}

        with mlflow.start_run():
            mlflow.log_params({
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "max_epochs": max_epochs,
                "patience": patience,
            })
            for epoch in range(max_epochs):
                train_loss = self.train_epoch(train_loader)
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                logger.info("epoch_complete", extra={"epoch": epoch, "train_loss": train_loss})

                val_loss = train_loss
                if val_loader is not None:
                    val_loss = self._evaluate_loss(val_loader)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        logger.info("early_stopping_triggered", extra={"epoch": epoch})
                        break

                history = {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch}

        return history

    @torch.no_grad()
    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for user_batch, item_batch, negative_batch in dataloader:
            user_batch = {k: v.to(self.device) for k, v in user_batch.items()}
            item_batch = {k: v.to(self.device) for k, v in item_batch.items()}
            negative_batch = {k: v.to(self.device) for k, v in negative_batch.items()}

            user_emb = self.model.encode_user(**user_batch)
            pos_item_emb = self.model.encode_item(**item_batch)
            batch_size, num_neg = negative_batch["item_ids"].shape
            flat_neg = {k: v.reshape(-1, *v.shape[2:]) for k, v in negative_batch.items()}
            neg_emb = self.model.encode_item(**flat_neg).reshape(batch_size, num_neg, -1)

            loss = self.model.sampled_softmax_loss(user_emb, pos_item_emb, neg_emb)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)
