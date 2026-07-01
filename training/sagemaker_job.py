"""SageMaker training job launcher with spot instance support.

Launches training/trainer.py as a remote PyTorch Estimator job on a single
A10G GPU instance (blueprint spec: ~4 hours for the weekly full retrain).
Guarded import of `sagemaker` — this module is only exercised in an AWS
environment with credentials configured; local dev never needs to import it.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def launch_training_job(
    s3_input_path: str,
    s3_output_path: str,
    role_arn: str,
    instance_type: str = "ml.g5.xlarge",
    use_spot_instances: bool = True,
    max_run_seconds: int = 4 * 3600,
    max_wait_seconds: int = 6 * 3600,
) -> str:
    from sagemaker.pytorch import PyTorch

    estimator = PyTorch(
        entry_point="trainer.py",
        source_dir="training",
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.2",
        py_version="py310",
        use_spot_instances=use_spot_instances,
        max_run=max_run_seconds,
        max_wait=max_wait_seconds if use_spot_instances else None,
        output_path=s3_output_path,
        hyperparameters={"epochs": 20, "batch-size": 256, "lr": 1e-3},
    )
    estimator.fit({"training": s3_input_path})
    job_name = estimator.latest_training_job.name
    logger.info("sagemaker_job_launched", extra={"job_name": job_name, "spot": use_spot_instances})
    return job_name
