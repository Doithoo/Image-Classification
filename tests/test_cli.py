"""Command-line parsing and actionable failure behavior."""

from __future__ import annotations

import argparse

import pytest

from garbage_classifier import cli
from garbage_classifier.data.manifest import ManifestError


@pytest.mark.parametrize(
    ("target_factory", "message"),
    [
        (lambda tmp_path: tmp_path / "missing", "input path does not exist"),
        (lambda tmp_path: tmp_path / "empty", "no supported images found"),
        (lambda tmp_path: tmp_path / "notes.txt", "unsupported image file"),
        (lambda tmp_path: tmp_path / "documents", "no supported images found"),
    ],
)
def test_predict_rejects_invalid_input_paths(tmp_path, monkeypatch, capsys, target_factory, message):
    target = target_factory(tmp_path)
    if target.name in {"empty", "documents"}:
        target.mkdir()
    if target.name == "notes.txt":
        target.write_text("not an image", encoding="utf-8")
    if target.name == "documents":
        (target / "notes.txt").write_text("not an image", encoding="utf-8")

    class PredictorMustNotLoad:
        def __init__(self, *args, **kwargs):
            raise AssertionError("input must be validated before loading a checkpoint")

    monkeypatch.setattr("garbage_classifier.inference.Predictor", PredictorMustNotLoad)

    result = cli.main(["predict", "--checkpoint", "model.pt", "--image", str(target)])

    assert result == 2
    assert capsys.readouterr().err.startswith(f"error: {message}")


@pytest.mark.parametrize(
    "argv",
    [
        ["predict", "--checkpoint", "model.pt", "--image", "image.jpg", "--top-k", "0"],
        ["predict", "--checkpoint", "model.pt", "--image", "image.jpg", "--top-k", "-1"],
        ["explain", "--checkpoint", "model.pt", "--image", "image.jpg", "--class-idx", "-1"],
        ["explain", "--checkpoint", "model.pt", "--image", "image.jpg", "--alpha", "-0.1"],
        ["explain", "--checkpoint", "model.pt", "--image", "image.jpg", "--alpha", "1.1"],
    ],
)
def test_numeric_cli_arguments_are_bounded(argv):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)

    assert exc_info.value.code == 2


def test_explain_rejects_class_outside_checkpoint_classes(tmp_path, monkeypatch, capsys):
    image = tmp_path / "image.jpg"
    image.touch()

    class FakePredictor:
        class_names = ["paper", "plastic"]

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("garbage_classifier.inference.Predictor", FakePredictor)

    result = cli.main(["explain", "--checkpoint", "model.pt", "--image", str(image), "--class-idx", "2"])

    assert result == 2
    assert "error: class index 2 is outside the valid range [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing checkpoint"),
        ManifestError("invalid manifest"),
        ValueError("invalid value"),
        KeyError("unknown model"),
    ],
)
def test_main_formats_expected_command_errors(monkeypatch, capsys, error):
    def fail(_args: argparse.Namespace) -> int:
        raise error

    monkeypatch.setattr(cli, "cmd_bench", fail)

    assert cli.main(["bench"]) == 2
    assert capsys.readouterr().err == f"error: {error}\n"


@pytest.mark.parametrize("argv", [["--debug", "bench"], ["bench", "--debug"]])
def test_debug_reraises_expected_command_errors(monkeypatch, argv):
    error = ValueError("invalid value")

    def fail(_args: argparse.Namespace) -> int:
        raise error

    monkeypatch.setattr(cli, "cmd_bench", fail)

    with pytest.raises(ValueError, match="invalid value"):
        cli.main(argv)


def test_main_does_not_swallow_argparse_system_exit():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["not-a-command"])

    assert exc_info.value.code == 2
