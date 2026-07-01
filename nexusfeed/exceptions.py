"""Custom exception hierarchy — every layer raises a subclass of NexusFeedError."""


class NexusFeedError(Exception):
    """Base exception for all NexusFeed errors."""


class FeatureStoreError(NexusFeedError):
    """Raised when a feature read/write fails (Redis or Postgres/S3)."""


class FeatureNotFoundError(FeatureStoreError):
    """Raised when a requested user/item feature does not exist yet (cold start)."""


class ModelError(NexusFeedError):
    """Raised for model load, inference, or training failures."""


class ModelNotLoadedError(ModelError):
    """Raised when inference is attempted before a model artifact is loaded."""


class RetrievalError(NexusFeedError):
    """Raised when FAISS index build or query fails."""


class IndexNotReadyError(RetrievalError):
    """Raised when a query hits an index that has not finished its initial build."""


class RankingError(NexusFeedError):
    """Raised when the re-ranking model fails to score candidates."""


class IngestionError(NexusFeedError):
    """Raised for Kafka producer/consumer failures."""


class EventValidationError(IngestionError):
    """Raised when an inbound event fails schema validation or anomaly checks."""


class DuplicateEventError(EventValidationError):
    """Raised when an event has already been seen (deduplication)."""


class ExperimentError(NexusFeedError):
    """Raised for A/B testing configuration or assignment failures."""


class ExperimentNotFoundError(ExperimentError):
    """Raised when a referenced experiment does not exist."""


class SampleRatioMismatchError(ExperimentError):
    """Raised when treatment/control split deviates from the configured ratio beyond tolerance."""


class RateLimitExceededError(NexusFeedError):
    """Raised by the rate limiter middleware; mapped to HTTP 429."""


class AuthenticationError(NexusFeedError):
    """Raised when an API key is missing, invalid, or lacks the required scope."""
