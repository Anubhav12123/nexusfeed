"""Feature versioning and metadata tracking.

Every feature computation run is tagged with a version identifier so that
models can record which feature version they were trained on — this is what
lets you answer "did this regression come from the model or the features?"
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from nexusfeed.config import Settings, get_settings


class FeatureVersionRecord(BaseModel):
    version: str
    description: str
    computed_at: datetime
    feature_names: list[str]


class FeatureRegistry:
    """In-memory registry for local/dev use; backed by a `feature_versions`
    table in Postgres in production (see db/models.py ModelVersion for the
    analogous pattern applied to models).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._versions: dict[str, FeatureVersionRecord] = {}

    def register(self, version: str, description: str, feature_names: list[str]) -> FeatureVersionRecord:
        record = FeatureVersionRecord(
            version=version,
            description=description,
            computed_at=datetime.utcnow(),
            feature_names=feature_names,
        )
        self._versions[version] = record
        return record

    def get(self, version: str) -> FeatureVersionRecord | None:
        return self._versions.get(version)

    def current(self) -> str:
        return self.settings.current_feature_version

    def list_versions(self) -> list[FeatureVersionRecord]:
        return sorted(self._versions.values(), key=lambda r: r.computed_at)
