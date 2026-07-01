"""Pydantic Settings — all runtime config sourced from env vars with validation."""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "nexusfeed"
    env: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key_header: str = "X-API-Key"
    valid_api_keys: str = "dev-key-local-only"  # comma separated, use a secrets manager in prod
    rate_limit_events_per_second: int = 100
    rate_limit_window_seconds: int = 1

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_user_events: str = "user-events"
    kafka_topic_item_events: str = "item-events"
    kafka_topic_model_updates: str = "model-updates"
    kafka_topic_impressions: str = "impressions"
    kafka_consumer_group: str = "nexusfeed-feature-pipeline"
    kafka_replication_factor: int = 3
    kafka_partitions_user_events: int = 24
    kafka_batch_size: int = 16384
    kafka_linger_ms: int = 5
    kafka_min_insync_replicas: int = 2

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_embedding_ttl_seconds: int = 3600
    redis_recent_items_ttl_seconds: int = 86400
    redis_experiment_ttl_seconds: int = 604800
    redis_freshness_ttl_seconds: int = 300
    redis_trending_ttl_seconds: int = 900

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://nexusfeed:nexusfeed@localhost:5432/nexusfeed"
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # S3 / object storage (offline feature + model artifacts)
    s3_bucket: str = "nexusfeed-artifacts-local"
    s3_endpoint_url: str | None = None  # set for localstack/minio in local dev
    aws_region: str = "us-east-1"

    # Model / ML
    embedding_dim: int = 128
    user_sequence_length: int = 50
    faiss_top_k: int = 1000
    feed_default_n: int = 20
    # Lives under ./data (the mounted volume + non-root-writable directory),
    # not the container's working directory — /app itself is root-owned
    # since it's created during the Docker build before USER nexusfeed.
    model_registry_uri: str = "sqlite:///./data/mlflow.db"
    ann_index_path: str = "./data/faiss_index.bin"
    ranking_model_path: str = "./data/ranking_model.txt"
    data_dir: str = "./data"

    # Feature versioning
    current_feature_version: str = "v1"

    # A/B testing
    ab_min_detectable_effect: float = 0.02
    ab_statistical_power: float = 0.8
    bandit_epsilon: float = 0.10
    holdback_fraction: float = 0.05
    default_experiment_name: str = "two_tower_v1_vs_baseline"

    # Model version shown in API responses when the registry has no
    # production-tagged version yet (fresh install, before the first
    # `make train` run). Once a model is promoted via ModelRegistry, the API
    # reads the real version from there instead of this fallback.
    fallback_model_version: str = "two_tower_v1"
    ranking_model_name: str = "nexusfeed-two-tower"

    # Latency targets (ms) — used by observability alert thresholds
    feed_p99_target_ms: float = 50.0
    events_p99_target_ms: float = 10.0

    # Observability
    otel_exporter_endpoint: str = "http://localhost:4317"
    prometheus_multiproc_dir: str | None = None

    @property
    def api_keys(self) -> set[str]:
        return {k.strip() for k in self.valid_api_keys.split(",") if k.strip()}

    @property
    def kafka_bootstrap_list(self) -> list[str]:
        return [s.strip() for s in self.kafka_bootstrap_servers.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
