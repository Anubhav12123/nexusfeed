"""LightGBM re-ranker with ONNX export — Layer 5 of the blueprint.

GBDTs beat deep nets here on speed (2ms per 1000-item batch after ONNX
export), interpretability (native feature importances + SHAP-friendly), and
because cross-features (user-item dot product, category affinity, time
decay, diversity score) are exactly the tabular signal LightGBM excels at.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

RANKING_FEATURE_NAMES = [
    "user_item_dot_product",
    "item_freshness_score",
    "user_item_category_affinity",
    "time_decay",
    "diversity_score",
    "historical_ctr",
    "popularity_score",
]


class RankingModel:
    def __init__(self, booster: lgb.Booster | None = None) -> None:
        self.booster = booster

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        num_boost_round: int = 200,
        params: dict | None = None,
    ) -> "RankingModel":
        params = params or {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_pre_filter": False,
            "verbose": -1,
        }
        train_set = lgb.Dataset(features, label=labels, feature_name=RANKING_FEATURE_NAMES)
        self.booster = lgb.train(params, train_set, num_boost_round=num_boost_round)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("RankingModel has no trained booster loaded")
        return self.booster.predict(features)

    def save(self, path: str | Path) -> None:
        if self.booster is None:
            raise RuntimeError("nothing to save — train() first")
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "RankingModel":
        booster = lgb.Booster(model_file=str(path))
        return cls(booster=booster)

    def export_onnx(self, path: str | Path) -> None:
        """Export to ONNX for the 2ms-per-1000-item serving path.
        Requires `onnxmltools` — kept as an optional import so the base
        package doesn't force-install the full ONNX toolchain in dev.
        """
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        if self.booster is None:
            raise RuntimeError("train() before exporting")
        initial_types = [("input", FloatTensorType([None, len(RANKING_FEATURE_NAMES)]))]
        onnx_model = convert_lightgbm(self.booster, initial_types=initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())


class OnnxRankingModel:
    """Serving-side ONNX Runtime wrapper — this is what the API process
    actually loads, since ONNX Runtime inference is ~3x faster than the
    native LightGBM Python predict() call under the feed endpoint's budget.
    """

    def __init__(self, onnx_path: str | Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def score(self, features: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: features.astype(np.float32)})
        # LightGBM ONNX export returns (label, probability_map) for binary objective
        probs = outputs[1]
        return np.array([p[1] for p in probs], dtype=np.float32)
