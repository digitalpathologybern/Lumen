"""Resume-safe MLflow tracking for multi-window Lumen training."""

from __future__ import annotations

import json
import logging
import os
import socket
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger("lumen.mlflow")


def _flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping):
            flat.update(_flatten(value, key))
        elif isinstance(value, (list, tuple)):
            flat[key] = json.dumps(value)
        elif value is None:
            flat[key] = "null"
        else:
            flat[key] = value
    return flat


class MLflowTracker:
    """Thin manual logger that resumes one run across Slurm windows."""

    def __init__(self, mlflow_module, run_id: str, out_dir: Path) -> None:
        self.mlflow = mlflow_module
        self.run_id = run_id
        self.out_dir = out_dir

    @classmethod
    def start(cls, cfg, out_dir: Path, data_stats: Mapping[str, Any] | None = None):
        if not cfg.mlflow_enabled:
            return NullTracker()
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking is enabled but mlflow is not installed; "
                "install the environment.yml dependency"
            ) from exc

        tracking_uri = (os.environ.get("MLFLOW_TRACKING_URI")
                        or cfg.mlflow_tracking_uri)
        if not tracking_uri:
            db = (Path.cwd() / "outputs" / "mlflow.db").resolve()
            tracking_uri = f"sqlite:///{db}"
        mlflow.set_tracking_uri(tracking_uri)

        artifact_root = (Path.cwd() / "outputs" / "mlflow-artifacts").resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(cfg.mlflow_experiment)
        if experiment is None:
            experiment_id = client.create_experiment(
                cfg.mlflow_experiment, artifact_location=artifact_root.as_uri())
        else:
            experiment_id = experiment.experiment_id

        run_id_path = out_dir / "mlflow_run_id.txt"
        resumed = run_id_path.exists()
        if resumed:
            run_id = run_id_path.read_text().strip()
            run = mlflow.start_run(run_id=run_id)
        else:
            run = mlflow.start_run(experiment_id=experiment_id,
                                   run_name=cfg.run_name)
            run_id = run.info.run_id
            run_id_path.write_text(run_id + "\n")

        tracker = cls(mlflow, run_id, out_dir)
        mlflow.set_tags({
            "training.backend": "custom-pytorch-gradcache",
            "training.output_dir": str(out_dir.resolve()),
            "slurm.last_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "host.last": socket.gethostname(),
            "run.resumed": str(resumed).lower(),
        })
        if not resumed:
            params = _flatten(asdict(cfg))
            if data_stats:
                params.update(_flatten(dict(data_stats), "data_stats"))
            mlflow.log_params(params)
            mlflow.log_artifact(str(out_dir / "train_config.json"),
                                artifact_path="provenance")
        # Alembic's logging configuration can disable loggers that were created
        # before the first SQLite migration. Restore the training loggers so the
        # Slurm .log remains useful alongside MLflow.
        for logger_name in ("lumen.train", "lumen.mlflow",
                            "lumen.training.data"):
            logger = logging.getLogger(logger_name)
            logger.disabled = False
            logger.setLevel(logging.INFO)
        LOGGER.info("MLflow run %s (%s) -> %s", run_id,
                    "resumed" if resumed else "new", tracking_uri)
        return tracker

    def log_step(self, step: int, epoch: int, loss: float, lr: float,
                 grad_scale: float, gpu_allocated: float = 0.0,
                 gpu_reserved: float = 0.0) -> None:
        self._safe(self.mlflow.log_metrics, {
            "step.train_loss_running": float(loss),
            "step.learning_rate": float(lr),
            "step.epoch": float(epoch + 1),
            "step.grad_scale": float(grad_scale),
            "system.cuda_peak_allocated_gib": float(gpu_allocated),
            "system.cuda_peak_reserved_gib": float(gpu_reserved),
        }, step=step)

    def log_epoch(self, epoch: int, record: Mapping[str, Any]) -> None:
        metrics = {
            "epoch.train_loss": float(record["train_loss"]),
            "epoch.val_loss": float(record["val_loss"]),
            "epoch.val_accuracy": float(record["val_acc"]),
            "epoch.learning_rate_end": float(record["lr_end"]),
            "epoch.duration_seconds": float(record["secs"]),
        }
        self._safe(self.mlflow.log_metrics, metrics, step=epoch + 1)

    def log_checkpoint_metadata(self) -> None:
        for name in (
            "history.json", "checkpoint_best_model_cfg.json",
            "checkpoint_best_model.sha256", "checkpoint_last.sha256",
            "train_config.json", "mlflow_run_id.txt",
        ):
            path = self.out_dir / name
            if path.exists():
                self._safe(self.mlflow.log_artifact, str(path),
                           artifact_path="training")
        for checkpoint in ("checkpoint_best_model.pt", "checkpoint_last.pt"):
            path = self.out_dir / checkpoint
            if path.exists():
                self._safe(self.mlflow.set_tag,
                           f"checkpoint.{checkpoint}.path", str(path.resolve()))

    def finish(self) -> None:
        self.log_checkpoint_metadata()
        self.mlflow.end_run(status="FINISHED")

    @staticmethod
    def _safe(fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # tracking must not waste a GPU training window
            LOGGER.warning("MLflow logging failed: %s", exc)


class NullTracker:
    run_id = "disabled"

    def log_step(self, *args, **kwargs):
        pass

    def log_epoch(self, *args, **kwargs):
        pass

    def log_checkpoint_metadata(self, *args, **kwargs):
        pass

    def finish(self):
        pass
