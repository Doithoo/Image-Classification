"""Evaluation evidence uses checkpoint preprocessing and verified dataset identity."""

import json
from dataclasses import replace

import pytest
import torch
from PIL import Image

from garbage_classifier.config import load_config
from garbage_classifier.data.manifest import build_manifest, load_dataset_metadata
from garbage_classifier.evaluation.evaluate import EVALUATION_SCHEMA_VERSION, evaluate_checkpoint
from garbage_classifier.training.checkpoint import save_checkpoint


def _prepared_two_class_data(tmp_path):
    for class_index, name in enumerate(("a", "b")):
        directory = tmp_path / "data" / name
        directory.mkdir(parents=True)
        for index in range(10):
            Image.new("RGB", (24, 24), color=(class_index * 100, index * 10, 0)).save(directory / f"{index}.jpg")
    build_manifest(tmp_path / "data", tmp_path / "manifests", seed=3)


def test_evaluate_uses_checkpoint_preprocessing_and_publishes_confidence_evidence(tmp_path, monkeypatch):
    _prepared_two_class_data(tmp_path)
    metadata = load_dataset_metadata(tmp_path / "manifests")
    checkpoint_cfg = load_config(
        overrides={
            "data.data_dir": str(tmp_path / "data"),
            "data.manifest_dir": str(tmp_path / "manifests"),
            "data.image_size": 19,
            "data.resize_size": 23,
            "data.normalize_mean": [0.1, 0.2, 0.3],
            "data.normalize_std": [0.4, 0.5, 0.6],
            "model.name": "resnet18",
            "model.num_classes": 2,
            "train.batch_size": 7,
            "device": "cpu",
        }
    )
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model=torch.nn.Linear(2, 2),
        epoch=1,
        best_metric=0.5,
        cfg=checkpoint_cfg,
        class_names=list(metadata.classes),
        manifest_identity=metadata.identity,
    )
    captured = {}

    def fake_transform(data_cfg):
        captured["data_cfg"] = data_cfg
        return lambda _image: torch.zeros(3, data_cfg.image_size, data_cfg.image_size)

    class FakePredictor:
        def __init__(self, checkpoint_path, device, config_path=None):
            captured.update(checkpoint_path=checkpoint_path, device=device, config_path=config_path)

        def predict_probs(self, images, tta=False):
            captured["shape"] = tuple(images.shape)
            probabilities = torch.zeros(len(images), 2)
            probabilities[:, 0] = 0.9
            probabilities[:, 1] = 0.1
            return probabilities

    monkeypatch.setattr("garbage_classifier.evaluation.evaluate.build_eval_transform", fake_transform)
    monkeypatch.setattr("garbage_classifier.inference.predictor.Predictor", FakePredictor)
    runtime_cfg = replace(checkpoint_cfg, train=replace(checkpoint_cfg.train, batch_size=3))

    metrics = evaluate_checkpoint(checkpoint, runtime_cfg, output_dir=tmp_path / "output")

    report = json.loads((tmp_path / "output" / "evaluation.json").read_text())
    assert report["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert report["manifest_identity"] == metadata.identity
    assert report["metrics"] == metrics
    assert "nll" in metrics and "ece" in metrics and metrics["top_5_accuracy"] == 1.0
    assert (tmp_path / "output" / "predictions.csv").is_file()
    assert (tmp_path / "output" / "errors.csv").is_file()
    assert (tmp_path / "output" / "calibration.csv").is_file()
    assert captured["data_cfg"].image_size == 19
    assert captured["shape"][1:] == (3, 19, 19)


def test_evaluation_rejects_checkpoint_for_different_verified_dataset(tmp_path):
    _prepared_two_class_data(tmp_path)
    metadata = load_dataset_metadata(tmp_path / "manifests")
    cfg = load_config(
        overrides={
            "data.data_dir": str(tmp_path / "data"),
            "data.manifest_dir": str(tmp_path / "manifests"),
            "model.num_classes": 2,
        }
    )
    checkpoint = tmp_path / "wrong.pt"
    save_checkpoint(
        checkpoint,
        model=torch.nn.Linear(2, 2),
        epoch=1,
        best_metric=0.0,
        cfg=cfg,
        class_names=list(metadata.classes),
        manifest_identity="another-dataset",
    )
    from garbage_classifier.training.checkpoint import CheckpointCompatibilityError

    with pytest.raises(CheckpointCompatibilityError, match="manifest_identity"):
        evaluate_checkpoint(checkpoint, cfg)
