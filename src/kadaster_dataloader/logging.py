from typing import Any, Dict, Protocol

import mlflow
from loguru import logger


class ExperimentLogger(Protocol):
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None: ...

    def log_params(self, params: Dict[str, Any]) -> None: ...

    def log_artifact(self, local_path: str) -> None: ...


class ConsoleLogger:
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step} | {log_str}")

    def log_params(self, params: Dict[str, Any]) -> None:
        logger.info(f"Params: {params}")

    def log_artifact(self, local_path: str) -> None:
        logger.info(f"Artifact saved at: {local_path}")


class MLFlowLogger:
    def __init__(self, experiment_name: str = "kadaster_experiment"):
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()
        logger.info(f"MLflow run started: {self.run.info.run_id}")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def __del__(self):
        mlflow.end_run()


class CompositeLogger:
    def __init__(self, loggers: list[ExperimentLogger]):
        self.loggers = loggers

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for exp_logger in self.loggers:
            exp_logger.log_metrics(metrics, step)

    def log_params(self, params: Dict[str, Any]) -> None:
        for exp_logger in self.loggers:
            exp_logger.log_params(params)

    def log_artifact(self, local_path: str) -> None:
        for exp_logger in self.loggers:
            exp_logger.log_artifact(local_path)
