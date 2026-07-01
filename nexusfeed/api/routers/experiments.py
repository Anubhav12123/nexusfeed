"""Experiment management endpoints — GET /experiments/{user_id}, admin CRUD."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Body, Request

from nexusfeed.db.connection import get_db
from nexusfeed.db.repositories.experiment_repo import ExperimentRepository
from nexusfeed.experiments.experiment_manager import ExperimentManager

router = APIRouter(tags=["experiments"])


@router.get("/experiments/{user_id}")
async def get_experiment_assignment(user_id: UUID, request: Request):
    async for db in get_db():
        manager = ExperimentManager(ExperimentRepository(db), request.app.state.redis)
        experiment_name = request.app.state.settings.default_experiment_name
        try:
            assignment = await manager.get_assignment(user_id, experiment_name)
        except Exception:
            return {
                "user_id": str(user_id),
                "variant": "control",
                "bucket": None,
                "note": "no active experiment",
            }
        return assignment.model_dump()


@router.post("/experiments")
async def create_experiment(
    request: Request,
    name: str = Body(...),
    control_range: tuple[int, int] = Body((0, 50)),
    treatment_range: tuple[int, int] = Body((50, 100)),
    config: dict = Body(default_factory=dict),
):
    async for db in get_db():
        manager = ExperimentManager(ExperimentRepository(db), request.app.state.redis)
        experiment = await manager.create_experiment(name, control_range, treatment_range, config)
        await db.commit()
        return {"id": str(experiment.id), "name": experiment.name, "status": experiment.status}


@router.post("/experiments/{name}/stop")
async def stop_experiment(name: str, request: Request):
    async for db in get_db():
        manager = ExperimentManager(ExperimentRepository(db), request.app.state.redis)
        await manager.stop_experiment(name)
        await db.commit()
        return {"name": name, "status": "stopped"}
