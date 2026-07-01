"""PyTorch training loop entry point with MLflow tracking — CLI wrapper
around nexusfeed.models.train.Trainer for standalone/SageMaker execution.
"""
from __future__ import annotations

import argparse
import logging

from torch.utils.data import DataLoader, random_split

from nexusfeed.models.model_registry import ModelRegistry
from nexusfeed.models.train import Trainer
from nexusfeed.models.two_tower import TwoTowerModel
from training.data_loader import InteractionDataset, load_training_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NexusFeed two-tower model")
    parser.add_argument("--data", required=True, help="Path to interaction Parquet file")
    parser.add_argument("--num-users", type=int, required=True)
    parser.add_argument("--num-items", type=int, required=True)
    parser.add_argument("--num-categories", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="./data/two_tower_model.pt")
    args = parser.parse_args()

    df = load_training_dataframe(args.data)
    dataset = InteractionDataset(df, num_items=args.num_items)
    val_size = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = TwoTowerModel(
        num_users=args.num_users, num_items=args.num_items, num_categories=args.num_categories
    )
    trainer = Trainer(model, learning_rate=args.lr)
    history = trainer.fit(train_loader, val_loader, max_epochs=args.epochs)
    logger.info("training_complete", extra=history)

    import torch

    torch.save(model.state_dict(), args.output)
    logger.info("model_saved", extra={"path": args.output})

    from nexusfeed.config import get_settings

    settings = get_settings()
    registry = ModelRegistry(settings)
    registry.register(
        model_name=settings.ranking_model_name,
        artifact_uri=args.output,
        metrics=history,
        feature_version=settings.current_feature_version,
    )


if __name__ == "__main__":
    main()
