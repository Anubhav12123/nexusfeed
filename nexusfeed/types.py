"""All Pydantic models used across the system. Every other module imports from here."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    VIEW_FULL = "view_full"
    VIEW_SCROLL_PAST = "view_scroll_past"
    CLICK = "click"
    DWELL = "dwell"
    SHARE = "share"
    SAVE = "save"
    SKIP = "skip"
    REPORT = "report"
    HIDE = "hide"


# Signal strength weights used at training time — mirrors the blueprint's event table.
EVENT_SIGNAL_WEIGHT: dict[EventType, float] = {
    EventType.VIEW_FULL: 1.0,
    EventType.VIEW_SCROLL_PAST: -0.25,
    EventType.CLICK: 1.5,
    EventType.DWELL: 1.25,
    EventType.SHARE: 3.0,
    EventType.SAVE: 3.0,
    EventType.SKIP: -1.0,
    EventType.REPORT: -3.0,
    EventType.HIDE: -3.0,
}

NEGATIVE_EVENTS = {
    EventType.VIEW_SCROLL_PAST,
    EventType.SKIP,
    EventType.REPORT,
    EventType.HIDE,
}


class Event(BaseModel):
    """A single behavioral event ingested via POST /events."""

    event_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    item_id: UUID
    event_type: EventType
    dwell_ms: int | None = None
    session_id: UUID | None = None
    device: str | None = None
    client_timestamp: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def is_negative(self) -> bool:
        return self.event_type in NEGATIVE_EVENTS

    def signal_weight(self) -> float:
        return EVENT_SIGNAL_WEIGHT[self.event_type]


class UserProfile(BaseModel):
    user_id: UUID
    embedding: list[float] | None = None
    recent_item_ids: list[UUID] = Field(default_factory=list)
    interaction_count: int = 0
    device: str | None = None
    location: str | None = None
    onboarding_categories: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def is_cold_start(self, threshold: int = 10) -> bool:
        return self.interaction_count < threshold


class ItemProfile(BaseModel):
    item_id: UUID
    category: str
    content_text: str | None = None
    embedding: list[float] | None = None
    freshness_score: float = 1.0
    popularity_score: float = 0.0
    impression_count: int = 0
    historical_ctr: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def is_cold_start(self, threshold: int = 100) -> bool:
        return self.impression_count < threshold


class FeatureVersion(BaseModel):
    version: str
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    description: str | None = None


class ScoredItem(BaseModel):
    # str, not UUID: FAISS/Redis/category/embedding lookups throughout the
    # retrieval and ranking layers key their dicts and sets by plain string
    # item ids. Typing this as UUID would make pydantic silently coerce the
    # value to a uuid.UUID object, and every downstream `dict.get(item_id)` /
    # `item_id in seen_items` check against a str-keyed collection would then
    # silently miss (wrong type, no error) — quietly disabling diversity,
    # freshness boost, and seen-item penalty.
    item_id: str
    score: float
    category: str | None = None
    is_trending: bool = False
    freshness_boosted: bool = False
    explanation: dict[str, float] | None = None


class FeedRequest(BaseModel):
    user_id: UUID
    n: int = 20
    device: str | None = None


class FeedItem(BaseModel):
    item_id: str  # see ScoredItem.item_id — same reasoning
    score: float
    rank: int
    category: str | None = None
    is_trending: bool = False
    explanation: str | None = None


class FeedResponse(BaseModel):
    user_id: UUID
    items: list[FeedItem]
    model_version: str
    experiment_bucket: int | None = None
    experiment_variant: str | None = None
    latency_ms: float
    request_id: str | None = None


class EventIngestResponse(BaseModel):
    event_id: UUID
    status: str = "accepted"


class FeedbackRequest(BaseModel):
    user_id: UUID
    item_id: UUID
    feedback_type: EventType
    reason: str | None = None


class SimilarItemsResponse(BaseModel):
    item_id: UUID
    similar_items: list[FeedItem]


class ExperimentConfig(BaseModel):
    experiment_id: UUID = Field(default_factory=uuid4)
    name: str
    control_range: tuple[int, int]
    treatment_range: tuple[int, int]
    config: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None


class ExperimentAssignment(BaseModel):
    user_id: UUID
    experiment_name: str
    bucket: int
    variant: str  # "control" | "treatment" | "holdback"


class ExperimentMetrics(BaseModel):
    experiment_name: str
    variant: str
    impressions: int = 0
    clicks: int = 0
    shares: int = 0
    dwell_ms_total: int = 0
    returns: int = 0

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0


class SignificanceResult(BaseModel):
    metric: str
    control_value: float
    treatment_value: float
    relative_lift: float
    p_value: float
    is_significant: bool
    sample_ratio_ok: bool


class ModelVersion(BaseModel):
    model_name: str
    version: str
    artifact_uri: str
    metrics: dict[str, float]
    feature_version: str
    status: str = "staging"  # staging | production | retired | canary
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HealthStatus(BaseModel):
    status: str
    components: dict[str, str]
    version: str
