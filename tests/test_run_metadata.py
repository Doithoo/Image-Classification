from datetime import datetime, timezone

import torch
import yaml

from garbage_classifier.training.metadata import build_run_metadata, write_run_metadata


def test_run_metadata_records_the_environment_and_elapsed_time(tmp_path, monkeypatch):
    monkeypatch.setattr("garbage_classifier.training.metadata.git_revision", lambda: "abc1234")
    monkeypatch.setattr("garbage_classifier.training.metadata.platform.platform", lambda: "test-platform")
    monkeypatch.setattr("garbage_classifier.training.metadata.platform.machine", lambda: "test-machine")
    started = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 8, 13, 12, 2, tzinfo=timezone.utc)

    metadata = build_run_metadata(
        torch.device("cpu"),
        started_at=started,
        finished_at=finished,
        elapsed_seconds=120.25,
    )
    write_run_metadata(tmp_path / "run.yaml", metadata)

    saved = yaml.safe_load((tmp_path / "run.yaml").read_text())
    assert saved["schema_version"] == 2
    assert saved["started_at"] == "2026-08-13T12:00:00+00:00"
    assert saved["finished_at"] == "2026-08-13T12:02:00+00:00"
    assert saved["elapsed_seconds"] == 120.25
    assert saved["python"]
    assert saved["pytorch"] == str(torch.__version__)
    assert saved["torchvision"]
    assert saved["platform"] == "test-platform"
    assert saved["device"] == "cpu"
    assert saved["accelerator"] == "test-machine"
    assert saved["git_revision"] == "abc1234"
