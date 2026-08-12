from pathlib import Path

import torch

from garbage_classifier.config import load_config, to_dict
from garbage_classifier.evaluation.evaluate import evaluate_checkpoint


def test_evaluate_uses_checkpoint_model_and_preprocessing_with_cli_runtime_overrides(tmp_path, monkeypatch):
    checkpoint_cfg = load_config(
        overrides={
            "data.image_size": 19,
            "data.resize_size": 23,
            "data.normalize_mean": [0.1, 0.2, 0.3],
            "data.normalize_std": [0.4, 0.5, 0.6],
            "model.name": "checkpoint-model",
        }
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": {},
            "config": to_dict(checkpoint_cfg),
            "class_names": ["a", "b"],
        },
        checkpoint,
    )
    cli_cfg = load_config(
        overrides={
            "data.data_dir": str(tmp_path / "images"),
            "data.manifest_dir": str(tmp_path / "manifests"),
            "data.image_size": 99,
            "data.resize_size": 101,
            "data.normalize_mean": [0.7, 0.7, 0.7],
            "data.normalize_std": [0.9, 0.9, 0.9],
            "data.num_workers": 0,
            "model.name": "cli-model",
            "train.batch_size": 7,
            "device": "cpu",
        }
    )
    captured = {}

    def fake_transform(data_cfg):
        captured["data_cfg"] = data_cfg
        return object()

    class FakeDataset:
        samples = [("sample.jpg", 0)]

        def __init__(self, manifest_path, root_dir, transform):
            captured.update(manifest_path=manifest_path, root_dir=root_dir, transform=transform)

        def __len__(self):
            return 1

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            captured["loader_kwargs"] = kwargs

        def __iter__(self):
            yield torch.zeros(1, 3, 19, 19), torch.tensor([0])

    class FakePredictor:
        def __init__(self, checkpoint_path, device):
            captured.update(predictor_checkpoint=checkpoint_path, predictor_device=device)
            self.cfg = checkpoint_cfg

        def predict_probs(self, images, tta=False):
            captured["tensor_shape"] = tuple(images.shape)
            return torch.tensor([[1.0, 0.0]])

    monkeypatch.setattr("garbage_classifier.evaluation.evaluate.build_eval_transform", fake_transform)
    monkeypatch.setattr("garbage_classifier.evaluation.evaluate.ImageClassificationDataset", FakeDataset)
    monkeypatch.setattr("garbage_classifier.evaluation.evaluate.torch.utils.data.DataLoader", FakeLoader)
    monkeypatch.setattr("garbage_classifier.inference.predictor.Predictor", FakePredictor)

    evaluate_checkpoint(checkpoint, cli_cfg, output_dir=tmp_path / "output")

    data_cfg = captured["data_cfg"]
    assert (data_cfg.image_size, data_cfg.resize_size) == (19, 23)
    assert data_cfg.normalize_mean == [0.1, 0.2, 0.3]
    assert data_cfg.normalize_std == [0.4, 0.5, 0.6]
    assert captured["manifest_path"] == Path(cli_cfg.data.manifest_dir) / "test.csv"
    assert captured["root_dir"] == Path(cli_cfg.data.data_dir)
    assert captured["loader_kwargs"]["batch_size"] == 7
    assert captured["loader_kwargs"]["num_workers"] == 0
    assert captured["predictor_device"] == "cpu"
