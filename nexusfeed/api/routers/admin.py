"""Admin/ops endpoints — extension beyond the blueprint minimum.

Exposes the state a recruiter demo or an on-call engineer would actually want
to poke at: trending items right now, active experiments and their live
metrics, and current model/index status. Backs the optional Streamlit demo
dashboard in scripts/demo.py.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from nexusfeed.db.connection import get_db
from nexusfeed.db.repositories.experiment_repo import ExperimentRepository
from nexusfeed.experiments.metrics_tracker import MetricsTracker
from nexusfeed.experiments.significance import evaluate_experiment
from nexusfeed.features.online_store import OnlineFeatureStore
from nexusfeed.types import ExperimentMetrics

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/trending")
async def trending(request: Request, limit: int = 20):
    store = OnlineFeatureStore(request.app.state.redis)
    items = await store.get_trending(limit=limit)
    return {"trending": [{"item_id": iid, "score": score} for iid, score in items]}


@router.get("/experiments/{name}/results")
async def experiment_results(name: str, request: Request):
    async for db in get_db():
        repo = ExperimentRepository(db)
        experiment = await repo.get_by_name(name)
        if experiment is None:
            return {"error": "experiment not found"}

        tracker = MetricsTracker(repo)
        metrics = await tracker.get_metrics(experiment.id)
        control = metrics.get("control", ExperimentMetrics(experiment_name=name, variant="control"))
        treatment = metrics.get("treatment", ExperimentMetrics(experiment_name=name, variant="treatment"))
        significance = evaluate_experiment(control, treatment)

        return {
            "experiment": name,
            "control": control.model_dump(),
            "treatment": treatment.model_dump(),
            "significance": significance.model_dump(),
        }


@router.get("/system-status")
async def system_status(request: Request):
    faiss_index = getattr(request.app.state, "faiss_index", None)
    return {
        "model_version": getattr(request.app.state, "model_version", "unknown"),
        "faiss_index_ready": bool(faiss_index and faiss_index.is_ready),
        "faiss_index_size": faiss_index.size if faiss_index else 0,
    }
