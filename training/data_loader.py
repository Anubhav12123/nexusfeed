"""Reads interaction Parquet from S3 (or local Postgres export) and builds
a PyTorch Dataset for two-tower training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from training.negative_sampling import sample_in_batch_negatives


class InteractionDataset(Dataset):
    """Expects a dataframe with columns:
    user_id, item_id, category_id, device_id, time_of_day, age, label (1=click, 0=no-click),
    recent_item_sequence (list[int], padded/truncated to sequence_length),
    content_embedding (list[float], length content_embedding_dim),
    freshness_score, historical_ctr
    """

    def __init__(self, df: pd.DataFrame, num_items: int, sequence_length: int = 50, negative_ratio: int = 4):
        self.df = df.reset_index(drop=True)
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.negative_ratio = negative_ratio

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        user_batch = {
            "user_ids": torch.tensor(row["user_id"], dtype=torch.long),
            "device_ids": torch.tensor(row["device_id"], dtype=torch.long),
            "time_of_day": torch.tensor(row["time_of_day"], dtype=torch.long),
            "age": torch.tensor(row["age"], dtype=torch.float32),
            "recent_item_sequence": torch.tensor(
                _pad_sequence(row["recent_item_sequence"], self.sequence_length), dtype=torch.long
            ),
        }
        item_batch = {
            "item_ids": torch.tensor(row["item_id"], dtype=torch.long),
            "category_ids": torch.tensor(row["category_id"], dtype=torch.long),
            "content_embedding": torch.tensor(row["content_embedding"], dtype=torch.float32),
            "freshness_score": torch.tensor(row["freshness_score"], dtype=torch.float32),
            "historical_ctr": torch.tensor(row["historical_ctr"], dtype=torch.float32),
        }

        positive_idx = np.array([row["item_id"]])
        negatives = sample_in_batch_negatives(positive_idx, self.num_items, self.negative_ratio)[0]
        content_dim = len(row["content_embedding"])
        negative_batch = {
            "item_ids": torch.tensor(negatives, dtype=torch.long),
            "category_ids": torch.zeros(self.negative_ratio, dtype=torch.long),
            "content_embedding": torch.zeros(self.negative_ratio, content_dim, dtype=torch.float32),
            "freshness_score": torch.zeros(self.negative_ratio, dtype=torch.float32),
            "historical_ctr": torch.zeros(self.negative_ratio, dtype=torch.float32),
        }
        return user_batch, item_batch, negative_batch


def _pad_sequence(sequence: list[int], length: int) -> list[int]:
    sequence = list(sequence)[-length:]
    return [0] * (length - len(sequence)) + sequence


def load_training_dataframe(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)
