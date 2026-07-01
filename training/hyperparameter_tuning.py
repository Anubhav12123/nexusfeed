"""Optuna-based HPO for embedding dim, learning rate, batch size."""
from __future__ import annotations

import logging

import optuna
from torch.utils.data import DataLoader, random_split

from nexusfeed.models.train import Trainer
from nexusfeed.models.two_tower import TwoTowerModel
from training.data_loader import InteractionDataset, load_training_dataframe

logger = logging.getLogger(__name__)


def build_objective(data_path: str, num_users: int, num_items: int, num_categories: int):
    df = load_training_dataframe(data_path)

    def objective(trial: optuna.Trial) -> float:
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        dataset = InteractionDataset(df, num_items=num_items)
        val_size = max(1, int(0.1 * len(dataset)))
        train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = TwoTowerModel(
            num_users=num_users, num_items=num_items, num_categories=num_categories, embedding_dim=embedding_dim
        )
        trainer = Trainer(model, learning_rate=learning_rate)
        history = trainer.fit(train_loader, val_loader, max_epochs=5, patience=2)
        return history.get("val_loss", history.get("train_loss", float("inf")))

    return objective


def run_study(data_path: str, num_users: int, num_items: int, num_categories: int, n_trials: int = 20) -> optuna.Study:
    study = optuna.create_study(direction="minimize", study_name="nexusfeed-two-tower-hpo")
    study.optimize(build_objective(data_path, num_users, num_items, num_categories), n_trials=n_trials)
    logger.info("hpo_complete", extra={"best_params": study.best_params, "best_value": study.best_value})
    return study
