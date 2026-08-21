"""CPU integration tests for verified training, resume and evaluation evidence."""

from dataclasses import replace

import pytest
import torch
from PIL import Image

from garbage_classifier.config import dump_config, load_config
from garbage_classifier.data.manifest import build_manifest
from garbage_classifier.inference import Predictor
from garbage_classifier.models.registry import available_models, create_model, get_num_parameters
from garbage_classifier.training.checkpoint import load_checkpoint
from garbage_classifier.training.train import train_from_config


def _synthetic_dataset(tmp_path, per_class: dict[str, int] | None = None) -> None:
    for class_index, (class_name, count) in enumerate((per_class or {"a": 6, "b": 6, "c": 6}).items()):
        directory = tmp_path / "data" / class_name
        directory.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (32, 32), color=(index * 40 % 255, class_index * 40, 200)).save(
                directory / f"{class_name}{index}.jpg"
            )


def _tiny_config(tmp_path, *, epochs: int = 2):
    _synthetic_dataset(tmp_path)
    build_manifest(tmp_path / "data", tmp_path / "manifests", seed=1)
    return load_config(
        overrides={
            "data.data_dir": str(tmp_path / "data"),
            "data.manifest_dir": str(tmp_path / "manifests"),
            "data.num_workers": 0,
            "data.pin_memory": False,
            "model.name": "mobilenetv3_small_100",
            "model.num_classes": 3,
            "model.pretrained": False,
            "train.epochs": epochs,
            "train.batch_size": 4,
            "train.amp": False,
            "train.early_stop_patience": 0,
            "device": "cpu",
            "output_dir": str(tmp_path / "artifacts"),
            "run_name": "smoke",
        }
    )


def test_training_requires_verified_data_and_matching_class_assertion(tmp_path):
    cfg = load_config(
        overrides={
            "model.num_classes": 3,
            "data.manifest_dir": str(tmp_path / "missing"),
            "output_dir": str(tmp_path / "artifacts"),
        }
    )
    with pytest.raises(Exception, match="dataset.yaml"):
        train_from_config(cfg)

    cfg = _tiny_config(tmp_path)
    mismatched = replace(cfg, model=replace(cfg.model, num_classes=2))
    with pytest.raises(Exception, match="configured 2, prepared data requires 3"):
        train_from_config(mismatched)


def test_dry_run_resolves_classes_without_writing_run_artifacts(tmp_path):
    cfg = _tiny_config(tmp_path)
    run_dir = train_from_config(cfg, dry_run=True)

    assert not run_dir.exists()


@pytest.mark.parametrize("name", ["resnet18", "mobilenetv3_small_100", "efficientnet_b0", "convnext_tiny"])
def test_selected_registry_models_have_logits_contract(name):
    model = create_model(name, num_classes=6, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 6)
    assert get_num_parameters(model) > 0


def test_registry_lists_stable_models_without_constructing_weights():
    assert "resnet18" in available_models()
    assert "tv_resnet50" in available_models()
    with pytest.raises(KeyError):
        create_model("no_such_model", num_classes=6)


def test_train_evaluate_predict_and_safe_resume_roundtrip(tmp_path):
    cfg = _tiny_config(tmp_path)
    run_dir = train_from_config(cfg)
    checkpoint = run_dir / "best.pt"
    payload = load_checkpoint(checkpoint)

    assert payload["manifest_identity"]
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "run.yaml").is_file()
    assert not (run_dir / "evaluation").exists()
    with pytest.raises(FileExistsError, match="run directory already exists"):
        train_from_config(cfg)

    from garbage_classifier.evaluation.evaluate import evaluate_checkpoint

    metrics = evaluate_checkpoint(checkpoint, cfg, split="test")
    evidence = run_dir / "evaluation" / "test"
    assert metrics["top_5_accuracy"] == 1.0
    assert (evidence / "evaluation.json").is_file()
    assert (evidence / "predictions.csv").is_file()
    assert (evidence / "errors.csv").is_file()
    assert (evidence / "per_class.csv").is_file()
    with pytest.raises(FileExistsError, match="non-empty"):
        evaluate_checkpoint(checkpoint, cfg, split="test")
    evaluate_checkpoint(checkpoint, cfg, split="test", overwrite=True)

    predictor = Predictor(checkpoint, device="cpu")
    image = next((tmp_path / "data" / "a").glob("*.jpg"))
    assert len(predictor.predict_path(image, top_k=3)) == 3

    with pytest.raises(ValueError, match="resume requires last.pt"):
        train_from_config(cfg, resume=str(checkpoint))
    with (run_dir / "metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("3,9,9,0,0,0\n")
    assert train_from_config(cfg, resume=str(run_dir / "last.pt")) == run_dir
    assert "\n3," not in (run_dir / "metrics.csv").read_text(encoding="utf-8")


def test_checkpoint_resume_rejects_changed_data_identity(tmp_path):
    cfg = _tiny_config(tmp_path, epochs=1)
    run_dir = train_from_config(cfg)
    Image.new("RGB", (32, 32), color="white").save(tmp_path / "data" / "a" / "changed.jpg")
    with pytest.raises(Exception, match="source image checksum mismatch"):
        train_from_config(cfg, resume=str(run_dir / "last.pt"))


def test_dump_config_accepts_path_values(tmp_path):
    cfg = _tiny_config(tmp_path, epochs=1)
    path = tmp_path / "config.yaml"
    dump_config(cfg, path)
    assert load_config(path).data.data_dir == tmp_path / "data"
