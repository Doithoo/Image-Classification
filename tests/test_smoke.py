"""Smoke tests: model registry, forward passes, full train/eval/predict loop.

These run on CPU with tiny synthetic data so they are fast and CI-friendly.
"""

import pytest
import torch
from PIL import Image

from garbage_classifier.config import dump_config, load_config
from garbage_classifier.data.manifest import build_manifest
from garbage_classifier.inference import Predictor
from garbage_classifier.models.registry import available_models, create_model, get_num_parameters


def _synthetic_dataset(tmp_path, per_class: dict[str, int] | None = None) -> None:
    per_class = per_class or {"a": 6, "b": 6, "c": 6}
    for cls, n in per_class.items():
        d = tmp_path / "data" / cls
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (32, 32), color=(i * 40 % 255, 10, 200)).save(d / f"{cls}{i}.jpg")


@pytest.mark.parametrize(
    "name",
    [
        "resnet18",
        "mobilenetv3_small_100",
        "efficientnet_b0",
        "convnext_tiny",
        "swin_tiny_patch4_window7_224",
        "vit_base_patch16_224",
    ],
)
def test_model_forward(name):
    model = create_model(name, num_classes=6, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 6)
    assert get_num_parameters(model) > 0


def test_registry_exposes_models():
    names = available_models()
    assert "resnet50" in names
    assert "tv_resnet50" in names


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        create_model("no_such_model", num_classes=6)


def _tiny_train(tmp_path):
    """Build a tiny CPU config, dump it to YAML and return the config path."""
    _synthetic_dataset(tmp_path)
    build_manifest(tmp_path / "data", tmp_path / "manifests", seed=1)

    cfg = load_config(
        overrides={
            "data.manifest_dir": str(tmp_path / "manifests"),
            "data.num_workers": 0,
            "data.pin_memory": False,
            "model.name": "mobilenetv3_small_100",
            "model.pretrained": False,
            "train.epochs": 2,
            "train.batch_size": 4,
            "train.amp": False,
            "train.early_stop_patience": 5,
            "device": "cpu",  # tests must be CPU-only (CI is CPU)
            "output_dir": str(tmp_path / "artifacts"),
            "run_name": "smoke",
        }
    )
    cfg_path = tmp_path / "cfg.yaml"
    dump_config(cfg, cfg_path)
    return cfg_path


def _cli_args(config_path, **extra):
    class Args:
        pass

    a = Args()
    a.config = str(config_path)
    a.set = []
    a.device = "auto"
    a.output_dir = None
    a.resume = None
    a.image_size = None
    a.opset = 17
    a.no_verify = False
    for k, v in extra.items():
        setattr(a, k, v)
    return a


def test_train_eval_predict_roundtrip(tmp_path):
    cfg_path = _tiny_train(tmp_path)
    from garbage_classifier.cli import cmd_evaluate, cmd_predict, cmd_train

    assert cmd_train(_cli_args(cfg_path)) == 0

    ckpt = tmp_path / "artifacts" / "smoke" / "best.pt"
    assert ckpt.exists()
    assert (tmp_path / "artifacts" / "smoke" / "metrics.csv").exists()
    assert (tmp_path / "artifacts" / "smoke" / "config.yaml").exists()

    # evaluate on test split
    assert cmd_evaluate(_cli_args(cfg_path, checkpoint=str(ckpt), split="test", error_limit=5)) == 0
    assert (tmp_path / "artifacts" / "smoke" / "predictions.csv").exists()

    # resume from checkpoint (should be a no-op start at epoch 2)
    assert cmd_train(_cli_args(cfg_path, resume=str(ckpt))) == 0

    # predict a single image through the CLI command
    img = next(iter((tmp_path / "data" / "a").glob("*.jpg")))
    assert cmd_predict(_cli_args(cfg_path, checkpoint=str(ckpt), image=str(img), top_k=3)) == 0

    # export-onnx (verification is skipped without onnxruntime; onnx is optional)
    from garbage_classifier.cli import cmd_export_onnx

    assert cmd_export_onnx(_cli_args(cfg_path, checkpoint=str(ckpt), output=str(tmp_path / "model.onnx"))) == 0
    assert (tmp_path / "model.onnx").exists()
    assert (tmp_path / "model.onnx.meta.yaml").exists()


def test_predictor_from_checkpoint_is_self_contained(tmp_path):
    cfg_path = _tiny_train(tmp_path)
    from garbage_classifier.cli import cmd_train

    assert cmd_train(_cli_args(cfg_path)) == 0

    ckpt = tmp_path / "artifacts" / "smoke" / "best.pt"
    predictor = Predictor(ckpt, device="cpu")  # no config passed -> restored from checkpoint
    assert predictor.class_names == ["a", "b", "c"]
    img = Image.open(next(iter((tmp_path / "data" / "a").glob("*.jpg"))))
    top = predictor.predict(img, top_k=3)
    assert len(top) == 3
    assert all(isinstance(name, str) and 0.0 <= prob <= 1.0 for name, prob in top)
