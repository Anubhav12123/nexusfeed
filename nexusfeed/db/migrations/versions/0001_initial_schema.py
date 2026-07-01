"""initial schema — users, items, interactions (partitioned), experiments, model_versions

Revision ID: 0001
Revises:
Create Date: 2026-01-01
"""
from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")  # gen_random_uuid()

    op.execute(
        """
        CREATE TABLE users (
          id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
          metadata     JSONB DEFAULT '{}'
        )
        """
    )

    op.execute(
        """
        CREATE TABLE items (
          id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          category         VARCHAR(64) NOT NULL,
          content_text     TEXT,
          embedding        VECTOR(128),
          freshness_score  FLOAT NOT NULL DEFAULT 1.0,
          popularity_score FLOAT NOT NULL DEFAULT 0.0,
          impression_count INTEGER NOT NULL DEFAULT 0,
          historical_ctr   FLOAT NOT NULL DEFAULT 0.0,
          created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
          metadata         JSONB DEFAULT '{}'
        )
        """
    )
    op.execute("CREATE INDEX items_embedding_hnsw ON items USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX items_category_idx ON items (category, created_at DESC)")

    # Range-partitioned by created_at — created_at must be part of the PK for
    # native Postgres partitioning to work, hence the composite PK below.
    op.execute(
        """
        CREATE TABLE interactions (
          id           UUID NOT NULL DEFAULT gen_random_uuid(),
          user_id      UUID NOT NULL REFERENCES users(id),
          item_id      UUID NOT NULL REFERENCES items(id),
          event_type   VARCHAR(32) NOT NULL,
          dwell_ms     INTEGER,
          session_id   UUID,
          created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
          PRIMARY KEY (id, created_at)
        ) PARTITION BY RANGE (created_at)
        """
    )
    op.execute("CREATE INDEX interactions_user_idx ON interactions (user_id, created_at DESC)")
    op.execute("CREATE INDEX interactions_item_idx ON interactions (item_id, event_type)")

    # Pre-create partitions for a rolling 24-month window around "today" so
    # local/dev/demo data lands somewhere valid without an operator running
    # a separate partition-management job on day one.
    op.execute(
        """
        DO $$
        DECLARE
          start_date date := date_trunc('month', now())::date - interval '12 months';
          part_date date;
          part_name text;
        BEGIN
          FOR i IN 0..23 LOOP
            part_date := (start_date + (i || ' months')::interval)::date;
            part_name := 'interactions_' || to_char(part_date, 'YYYY_MM');
            EXECUTE format(
              'CREATE TABLE IF NOT EXISTS %I PARTITION OF interactions FOR VALUES FROM (%L) TO (%L)',
              part_name, part_date, (part_date + interval '1 month')::date
            );
          END LOOP;
        END $$;
        """
    )

    op.execute(
        """
        CREATE TABLE experiments (
          id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          name            VARCHAR(128) UNIQUE NOT NULL,
          control_low     INTEGER NOT NULL,
          control_high    INTEGER NOT NULL,
          treatment_low   INTEGER NOT NULL,
          treatment_high  INTEGER NOT NULL,
          config          JSONB NOT NULL DEFAULT '{}',
          status          VARCHAR(16) NOT NULL DEFAULT 'active',
          started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
          ended_at        TIMESTAMPTZ
        )
        """
    )

    op.execute(
        """
        CREATE TABLE experiment_results (
          id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          experiment_id    UUID NOT NULL REFERENCES experiments(id),
          variant          VARCHAR(16) NOT NULL,
          impressions      INTEGER NOT NULL DEFAULT 0,
          clicks           INTEGER NOT NULL DEFAULT 0,
          shares           INTEGER NOT NULL DEFAULT 0,
          dwell_ms_total   INTEGER NOT NULL DEFAULT 0,
          returns          INTEGER NOT NULL DEFAULT 0,
          updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE (experiment_id, variant)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE model_versions (
          id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          model_name       VARCHAR(64) NOT NULL,
          version          VARCHAR(32) NOT NULL,
          artifact_uri     TEXT NOT NULL,
          metrics          JSONB NOT NULL DEFAULT '{}',
          feature_version  VARCHAR(32),
          status           VARCHAR(16) NOT NULL DEFAULT 'staging',
          created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS model_versions")
    op.execute("DROP TABLE IF EXISTS experiment_results")
    op.execute("DROP TABLE IF EXISTS experiments")
    op.execute("DROP TABLE IF EXISTS interactions CASCADE")
    op.execute("DROP TABLE IF EXISTS items")
    op.execute("DROP TABLE IF EXISTS users")
