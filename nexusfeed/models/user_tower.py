"""User encoder — LSTM over the last 50 interactions plus static user features.

Input features: user ID embedding, age, location, device, time-of-day, and
the recent-interaction sequence (item ID embeddings fed through an LSTM so
the tower captures short-term intent, not just a static user profile).
"""
from __future__ import annotations

import torch
from torch import nn


class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        num_devices: int = 8,
        sequence_length: int = 50,
        lstm_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding_for_sequence = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.device_embedding = nn.Embedding(num_devices, 8)
        self.time_of_day_embedding = nn.Embedding(24, 8)

        self.sequence_lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True, num_layers=1
        )

        static_dim = embedding_dim + 8 + 8 + 1  # user_id + device + time_of_day + age (scalar)
        combined_dim = static_dim + lstm_hidden_dim

        self.projection = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        device_ids: torch.Tensor,
        time_of_day: torch.Tensor,
        age: torch.Tensor,
        recent_item_sequence: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.user_id_embedding(user_ids)
        device_emb = self.device_embedding(device_ids)
        tod_emb = self.time_of_day_embedding(time_of_day)
        age_feat = age.unsqueeze(-1).float()

        seq_emb = self.item_id_embedding_for_sequence(recent_item_sequence)
        _, (h_n, _) = self.sequence_lstm(seq_emb)
        sequence_repr = h_n[-1]

        combined = torch.cat([user_emb, device_emb, tod_emb, age_feat, sequence_repr], dim=-1)
        out = self.projection(combined)
        return nn.functional.normalize(out, p=2, dim=-1)
