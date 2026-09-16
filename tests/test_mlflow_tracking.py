import logging
from pathlib import Path

from mlflow.tracking import MlflowClient

from lumen.training.config import TrainConfig
from lumen.training.tracking import MLflowTracker


def test_mlflow_run_is_logged_and_resumed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "training-run"
    output.mkdir()
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    cfg = TrainConfig(
        mlflow_enabled=True,
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment="tracking-test",
        out_dir=str(output),
        run_name="resume-test",
    )
    cfg.to_json(output / "train_config.json")

    first = MLflowTracker.start(cfg, output, {"n_train": 8})
    assert logging.getLogger("lumen.train").disabled is False
    first.log_step(1, 0, 2.0, 1e-4, 65536.0, 3.0, 4.0)
    first.log_epoch(0, {
        "train_loss": 2.0, "val_loss": 1.5, "val_acc": 0.25,
        "lr_end": 1e-4, "secs": 12.0,
    })
    run_id = first.run_id
    first.finish()

    second = MLflowTracker.start(cfg, output, {"n_train": 8})
    assert second.run_id == run_id
    second.log_step(2, 0, 1.8, 2e-4, 65536.0, 3.5, 4.5)
    second.finish()

    run = MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["gradient_cache"] == "False"
    assert run.data.params["data_stats.n_train"] == "8"
    assert run.data.metrics["step.train_loss_running"] == 1.8
    assert run.data.metrics["epoch.val_loss"] == 1.5
    assert run.data.tags["run.resumed"] == "true"
    assert Path(output / "mlflow_run_id.txt").read_text().strip() == run_id
