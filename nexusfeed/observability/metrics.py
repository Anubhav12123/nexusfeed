"""Prometheus metric definitions — counters, histograms, gauges.

Feed endpoint: request latency histogram (p50/p95/p99), FAISS retrieval time,
ranking time, cache hit rate. Event pipeline: consumer lag, events/sec, error
rate. Model: prediction score distribution, feature freshness lag. Twelve
metrics total, matching the "12 Prometheus metrics" resume bullet.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0)

# 1. Feed endpoint end-to-end latency
FEED_LATENCY_SECONDS = Histogram(
    "nexusfeed_feed_latency_seconds",
    "GET /feed end-to-end latency",
    buckets=LATENCY_BUCKETS,
)

# 2. Events endpoint latency
EVENTS_LATENCY_SECONDS = Histogram(
    "nexusfeed_events_latency_seconds",
    "POST /events latency",
    buckets=LATENCY_BUCKETS,
)

# 3. FAISS retrieval time
FAISS_RETRIEVAL_SECONDS = Histogram(
    "nexusfeed_faiss_retrieval_seconds",
    "ANN retrieval time for top-K candidate generation",
    buckets=LATENCY_BUCKETS,
)

# 4. Ranking time
RANKING_SECONDS = Histogram(
    "nexusfeed_ranking_seconds",
    "LightGBM re-ranking time for candidate set",
    buckets=LATENCY_BUCKETS,
)

# 5. Redis feature fetch cache hit rate
CACHE_HITS = Counter("nexusfeed_cache_hits_total", "Online feature store cache hits", ["store"])
CACHE_MISSES = Counter("nexusfeed_cache_misses_total", "Online feature store cache misses", ["store"])

# 6. Events processed per second (rate derived from this counter)
EVENTS_CONSUMED = Counter(
    "nexusfeed_events_consumed_total", "Events consumed by the feature pipeline", ["event_type"]
)

# 7. Kafka consumer lag
FEATURE_PIPELINE_LAG_SECONDS = Gauge(
    "nexusfeed_feature_pipeline_lag_seconds", "Age of the most recently processed event"
)

# 8. Ingestion error rate
INGESTION_ERRORS = Counter("nexusfeed_ingestion_errors_total", "Event ingestion validation/routing errors", ["reason"])

# 9. Prediction score distribution
PREDICTION_SCORE = Histogram(
    "nexusfeed_prediction_score",
    "Distribution of ranking model relevance scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# 10. Requests in flight / concurrency gauge
IN_FLIGHT_REQUESTS = Gauge("nexusfeed_in_flight_requests", "Requests currently being served", ["route"])

# 11. Rate limiter rejections
RATE_LIMIT_REJECTIONS = Counter("nexusfeed_rate_limit_rejections_total", "Requests rejected by the rate limiter")

# 12. Experiment assignment counter (A/B health)
EXPERIMENT_ASSIGNMENTS = Counter(
    "nexusfeed_experiment_assignments_total", "User experiment bucket assignments", ["experiment", "variant"]
)

# Bonus: model canary comparison gauge (Addition 5)
CANARY_CTR_DELTA = Gauge("nexusfeed_canary_ctr_delta", "Canary model CTR minus baseline CTR")
CANARY_LATENCY_DELTA_MS = Gauge("nexusfeed_canary_latency_delta_ms", "Canary model p99 latency minus baseline p99 (ms)")
