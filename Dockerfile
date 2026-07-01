# Multi-stage: builder (install deps) + runtime (minimal image)
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime

# libgomp1 is the OpenMP runtime lightgbm's compiled extension dynamically
# links against (libgomp.so.1) — python:3.11-slim doesn't ship it, so without
# this the container builds fine but crashes on the first `import lightgbm`
# with "OSError: libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 nexusfeed
WORKDIR /app

COPY --from=builder /install /usr/local
COPY nexusfeed ./nexusfeed
COPY training ./training
COPY alembic.ini .

# The whole working directory must be writable by the non-root runtime user,
# not just ./data: MLflow's SqlAlchemyStore creates a default artifact root
# ("./mlruns") relative to the process cwd on first use, regardless of where
# the tracking-store sqlite file itself lives — so ./data alone isn't enough.
RUN mkdir -p /app/data && chown -R nexusfeed:nexusfeed /app

USER nexusfeed
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "nexusfeed.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
