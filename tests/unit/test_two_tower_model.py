import torch

from nexusfeed.models.item_tower import ItemTower
from nexusfeed.models.two_tower import TwoTowerModel
from nexusfeed.models.user_tower import UserTower


def _make_user_batch(batch_size: int, num_users: int, num_items: int, seq_len: int = 50):
    return {
        "user_ids": torch.randint(0, num_users, (batch_size,)),
        "device_ids": torch.randint(0, 4, (batch_size,)),
        "time_of_day": torch.randint(0, 24, (batch_size,)),
        "age": torch.randint(13, 65, (batch_size,)),
        "recent_item_sequence": torch.randint(0, num_items, (batch_size, seq_len)),
    }


def _make_item_batch(batch_size: int, num_items: int, num_categories: int, content_dim: int = 768):
    return {
        "item_ids": torch.randint(0, num_items, (batch_size,)),
        "category_ids": torch.randint(0, num_categories, (batch_size,)),
        "content_embedding": torch.randn(batch_size, content_dim),
        "freshness_score": torch.rand(batch_size),
        "historical_ctr": torch.rand(batch_size),
    }


def test_user_tower_output_shape_and_normalization():
    tower = UserTower(num_users=100, num_items=200, embedding_dim=32)
    batch = _make_user_batch(4, 100, 200)
    out = tower(**batch)
    assert out.shape == (4, 32)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-4)


def test_item_tower_output_shape_and_normalization():
    tower = ItemTower(num_items=200, num_categories=10, embedding_dim=32)
    batch = _make_item_batch(4, 200, 10)
    out = tower(**batch)
    assert out.shape == (4, 32)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-4)


def test_two_tower_forward_returns_scalar_logits():
    model = TwoTowerModel(num_users=100, num_items=200, num_categories=10, embedding_dim=32)
    user_batch = _make_user_batch(4, 100, 200)
    item_batch = _make_item_batch(4, 200, 10)
    logits = model(user_batch, item_batch)
    assert logits.shape == (4,)


def test_sampled_softmax_loss_is_finite_and_positive():
    model = TwoTowerModel(num_users=100, num_items=200, num_categories=10, embedding_dim=16)
    user_emb = model.encode_user(**_make_user_batch(4, 100, 200))
    item_batch = _make_item_batch(4, 200, 10)
    positive_emb = model.encode_item(**item_batch)

    negatives = _make_item_batch(4 * 4, 200, 10)
    negative_emb = model.encode_item(**negatives).reshape(4, 4, -1)

    loss = model.sampled_softmax_loss(user_emb, positive_emb, negative_emb)
    assert torch.isfinite(loss)
    assert loss.item() > 0
