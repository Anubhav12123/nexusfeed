"""MLflow model registry integration: register, promote, rollback.

Addition 5 (Model Canary Deployment) and the interview answer on zero-downtime
FAISS hot-swap both hinge on this registry being the single source of truth
for "what is currently in production" vs "what is staged/canary."
"""
from __future__ import annotations

import logging

import mlflow
from mlflow.tracking import MlflowClient

from nexusfeed.config import Settings, get_settings
from nexusfeed.types import ModelVersion

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        mlflow.set_tracking_uri(self.settings.model_registry_uri)
        self.client = MlflowClient()

    def register(
        self, model_name: str, artifact_uri: str, metrics: dict[str, float], feature_version: str
    ) -> ModelVersion:
        try:
            self.client.create_registered_model(model_name)
        except Exception:  # noqa: BLE001 - already exists
            pass
        mv = self.client.create_model_version(name=model_name, source=artifact_uri, run_id=None)
        for key, value in metrics.items():
            self.client.set_model_version_tag(model_name, mv.version, f"metric.{key}", str(value))
        self.client.set_model_version_tag(model_name, mv.version, "feature_version", feature_version)
        self.client.transition_model_version_stage(model_name, mv.version, "Staging")
        logger.info("model_registered", extra={"model_name": model_name, "version": mv.version})
        return ModelVersion(
            model_name=model_name,
            version=mv.version,
            artifact_uri=artifact_uri,
            metrics=metrics,
            feature_version=feature_version,
            status="staging",
        )

    def promote_to_production(self, model_name: str, version: str) -> None:
        """Full promotion — used after canary validation passes (Addition 5)."""
        self.client.transition_model_version_stage(
            model_name, version, "Production", archive_existing_versions=True
        )
        logger.info("model_promoted", extra={"model_name": model_name, "version": version})

    def promote_to_canary(self, model_name: str, version: str, traffic_pct: float = 0.05) -> None:
        self.client.set_model_version_tag(model_name, version, "canary_traffic_pct", str(traffic_pct))
        self.client.set_model_version_tag(model_name, version, "stage_detail", "canary")

    def rollback(self, model_name: str) -> str | None:
        """Rollback to the previous production version — mirrors `kubectl
        rollout undo`, triggered automatically when canary latency/CTR
        thresholds are breached (Addition 5).
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        archived = [v for v in versions if v.current_stage == "Archived"]
        if not archived:
            logger.warning("no_previous_version_to_rollback_to", extra={"model_name": model_name})
            return None
        previous = max(archived, key=lambda v: int(v.version))
        self.client.transition_model_version_stage(
            model_name, previous.version, "Production", archive_existing_versions=True
        )
        logger.warning("model_rolled_back", extra={"model_name": model_name, "version": previous.version})
        return previous.version

    def get_production_version(self, model_name: str) -> str | None:
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        return versions[0].version if versions else None
