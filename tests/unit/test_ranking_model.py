import numpy as np

from nexusfeed.models.ranking_model import RANKING_FEATURE_NAMES, RankingModel


def _synthetic_dataset(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.uniform(0, 1, size=(n, len(RANKING_FEATURE_NAMES))).astype(np.float32)
    # label correlates with the first feature (user_item_dot_product) so the
    # model has real signal to learn — this is a determinism check, not a
    # claim about real-world AUC.
    labels = (features[:, 0] + rng.normal(0, 0.05, n) > 0.5).astype(int)
    return features, labels


def test_train_and_score_roundtrip(tmp_path):
    features, labels = _synthetic_dataset()
    model = RankingModel().train(features, labels, num_boost_round=20)

    scores = model.score(features)
    assert scores.shape == (len(features),)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_save_and_load_roundtrip(tmp_path):
    features, labels = _synthetic_dataset()
    model = RankingModel().train(features, labels, num_boost_round=20)

    path = tmp_path / "model.txt"
    model.save(path)

    loaded = RankingModel.load(path)
    original_scores = model.score(features[:10])
    loaded_scores = loaded.score(features[:10])
    assert np.allclose(original_scores, loaded_scores)


def test_model_learns_meaningful_signal():
    features, labels = _synthetic_dataset(n=2000)
    train_features, train_labels = features[:1500], labels[:1500]
    test_features, test_labels = features[1500:], labels[1500:]

    model = RankingModel().train(train_features, train_labels, num_boost_round=50)
    scores = model.score(test_features)

    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(test_labels, scores) > 0.7
