# Contributing to NexusFeed

## Local setup

```bash
make install-dev       # installs deps + pre-commit hooks
cp .env.example .env
make docker-up          # Postgres, Redis, Kafka, Prometheus, Grafana, API
make migrate             # apply Alembic migrations
make seed                # synthetic data + FAISS index + LightGBM ranker
make demo                 # exercises the full request flow
```

## Running tests

```bash
make test               # unit tests, 85%+ coverage required
make test-integration    # requires docker-compose services running
make load-test           # Locust, targets 1000 RPS / p99 < 50ms
```

## Adding a new feature

1. Add/extend Pydantic types in `nexusfeed/types.py` first — every other
   module imports from here, so getting the shape right up front avoids
   downstream refactors.
2. Follow the existing layer boundaries: ingestion -> features -> models ->
   retrieval -> ranking -> experiments -> api. A router should never reach
   directly into another router's internals — go through the shared layer.
3. Add a unit test alongside the module (`tests/unit/test_<module>.py`).
   Anything touching Redis/Postgres/Kafka directly should also get an
   integration test gated behind `@pytest.mark.integration`.
4. Add or update the relevant Prometheus metric in
   `nexusfeed/observability/metrics.py` if you're adding a new hot path.
5. Run `make format lint test` before opening a PR — `pre-commit` runs the
   same checks automatically on commit.

## Submitting a PR

- Keep PRs scoped to one layer/feature at a time.
- Include the milestone/benchmark this change affects (e.g. "feed p99 stays
  under 50ms with N=50") if it's performance-sensitive.
- CI must be green (lint, mypy on `types.py`, 85%+ coverage) before merge.
