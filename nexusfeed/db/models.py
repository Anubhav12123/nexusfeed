"""ORM models: User, Item, Interaction, Experiment, ExperimentResult, ModelVersion.

Mirrors the schema in NexusFeed_Project_Blueprint.pdf section 4, including the
monthly range-partitioning strategy on `interactions` and the pgvector HNSW
index on `items.embedding`.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 128


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)


class Item(Base):
    __tablename__ = "items"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    freshness_score: Mapped[float] = mapped_column(Float, default=1.0)
    popularity_score: Mapped[float] = mapped_column(Float, default=0.0)
    impression_count: Mapped[int] = mapped_column(Integer, default=0)
    historical_ctr: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index("items_category_idx", "category", "created_at"),
        # HNSW vector index is created via raw SQL in the Alembic migration
        # (pgvector's index type isn't expressible through the ORM Index API).
    )


class Interaction(Base):
    """Range-partitioned by created_at at the database level (see migration
    0001) — each month gets its own partition so "last 30 days" queries only
    scan 1-2 partitions instead of the full table.
    """

    __tablename__ = "interactions"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    item_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("items.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    dwell_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), primary_key=True
    )

    __table_args__ = (
        Index("interactions_user_idx", "user_id", "created_at"),
        Index("interactions_item_idx", "item_id", "event_type"),
    )


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    control_low: Mapped[int] = mapped_column(Integer, nullable=False)
    control_high: Mapped[int] = mapped_column(Integer, nullable=False)
    treatment_low: Mapped[int] = mapped_column(Integer, nullable=False)
    treatment_high: Mapped[int] = mapped_column(Integer, nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="active")
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    results: Mapped[list["ExperimentResult"]] = relationship(back_populates="experiment")


class ExperimentResult(Base):
    """Per-variant rollup metrics for an experiment, refreshed by the metrics
    tracker so the significance module can query aggregates without scanning
    raw interaction rows on every request.
    """

    __tablename__ = "experiment_results"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("experiments.id"))
    variant: Mapped[str] = mapped_column(String(16), nullable=False)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    shares: Mapped[int] = mapped_column(Integer, default=0)
    dwell_ms_total: Mapped[int] = mapped_column(Integer, default=0)
    returns: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    experiment: Mapped[Experiment] = relationship(back_populates="results")

    __table_args__ = (UniqueConstraint("experiment_id", "variant", name="uq_experiment_variant"),)


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    feature_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="staging")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
