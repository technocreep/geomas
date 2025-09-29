import mlflow
from mlflow.tracking import MlflowClient

from geomas.core.logging import get_logger

logger = get_logger()


def _init_mlflow_logging(
        correct_model_name: str,
        tag: str
        ):
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    prefix = f"{tag}-" if tag else ""
    exp_name = f"{prefix}CPT-{correct_model_name}"
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        logger.info(f"Experiment {exp_name} not found. Creating...")
        exp_id = client.create_experiment(
        name=exp_name,
        artifact_location=f"s3://mlflow/experiments/{exp_name}"
    )
    else:
        logger.info(f"Experiment {exp_name} exists")
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_id=exp_id)
    mlflow.enable_system_metrics_logging()