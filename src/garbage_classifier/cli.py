"""Command-line interface (presentation layer only).

All command logic lives in domain modules (``data.prepare``, ``training.train``,
``evaluation.evaluate``, ``inference.*``); this module only maps CLI arguments to
those functions.

Commands (config overrides via dotted keys, e.g. ``--set train.lr 1e-4``):
    show-config    Print the fully resolved configuration without running work
    prepare-data   Generate portable train/valid/test manifests from class folders
    train          Train a model (config-driven, resumable, AMP, early stopping)
    evaluate       Evaluate a checkpoint on a split (full metrics + optional plot)
    predict        Predict a single image or a folder of images
    export-onnx    Export a checkpoint to ONNX
    explain        Grad-CAM heatmap for a single image
    bench          List registry models with params/FLOPs
    demo           Launch the Gradio web demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from . import __version__
from .config import ExperimentConfig, load_config, to_dict
from .data.manifest import ManifestError
from .data.prepare import prepare_data
from .evaluation.evaluate import evaluate_checkpoint
from .training.train import train_from_config
from .utils import setup_logging


# ---- config helpers ---------------------------------------------------------
def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument(
        "--set",
        nargs=2,
        action="append",
        default=[],
        metavar=("KEY", "VALUE"),
        help="override config, dotted key, e.g. --set train.lr 1e-4",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--output-dir", type=str, default=None, help="artifacts root (default: config)")


def _parse_set(overrides: list[list[str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in overrides:
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _add_debug_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="show tracebacks for command errors",
    )


def _resolve_cfg(args: argparse.Namespace) -> ExperimentConfig:
    cfg = load_config(args.config, overrides=_parse_set(args.set))
    if args.device != "auto":
        cfg.device = args.device
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    return cfg


# ---- commands ---------------------------------------------------------------
def cmd_show_config(args: argparse.Namespace) -> int:
    """Print exactly the configuration that config-driven commands would use."""
    import yaml

    print(yaml.safe_dump(to_dict(_resolve_cfg(args)), sort_keys=False), end="")
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    prepare_data(
        data_dir=args.data_dir or cfg.data.data_dir,
        manifest_dir=cfg.data.manifest_dir,
        split_ratios=cfg.data.split_ratios,
        seed=cfg.data.seed,
        strict=args.strict,
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    setup_logging(cfg.log_level)
    train_from_config(cfg, resume=args.resume, dry_run=args.dry_run, run_name=cfg.run_name)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    setup_logging(cfg.log_level)
    evaluate_checkpoint(
        checkpoint=args.checkpoint,
        cfg=cfg,
        split=args.split,
        tta=args.tta,
        plot=args.plot,
        error_limit=args.error_limit,
        output_dir=args.output_dir,
    )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .inference import Predictor

    target = Path(args.image)
    supported_suffixes = (".jpg", ".jpeg", ".png")
    if not target.exists():
        raise FileNotFoundError(f"input path does not exist: {target}")
    if target.is_file():
        if target.suffix.lower() not in supported_suffixes:
            raise ValueError(f"unsupported image file: {target}")
        targets = [target]
    elif target.is_dir():
        targets = sorted(
            path for path in target.iterdir() if path.is_file() and path.suffix.lower() in supported_suffixes
        )
        if not targets:
            raise ValueError(f"no supported images found in directory: {target}")
    else:
        raise ValueError(f"input path is not a file or directory: {target}")

    predictor = Predictor(args.checkpoint, device=args.device, config_path=args.config)
    for path in targets:
        top = predictor.predict_path(path, top_k=args.top_k, tta=args.tta)
        ranked = ", ".join(f"{name}={prob:.3f}" for name, prob in top)
        print(f"{path} -> {ranked}")
    return 0


def cmd_export_onnx(args: argparse.Namespace) -> int:
    from .inference.export import export_onnx

    out = export_onnx(
        args.checkpoint,
        args.output,
        image_size=args.image_size,
        opset=args.opset,
        verify=not args.no_verify,
        device=args.device,
    )
    print(f"exported ONNX model to {out}")
    print(f"metadata sidecar: {out.with_suffix('.onnx.meta.yaml')}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from .inference import Predictor
    from .inference.gradcam import GradCAM, overlay_heatmap

    predictor = Predictor(args.checkpoint, device=args.device, config_path=args.config)
    if args.class_idx is not None and args.class_idx >= len(predictor.class_names):
        raise ValueError(
            f"class index {args.class_idx} is outside the valid range [0, {len(predictor.class_names) - 1}]"
        )
    image = Image.open(args.image)
    img_tensor = predictor.transform(image.convert("RGB")).to(predictor.device)

    cam_model = GradCAM(predictor.model)
    heatmap, class_idx = cam_model.generate(img_tensor, class_idx=args.class_idx, device=predictor.device)

    out = Path(args.output)
    overlay_heatmap(image, heatmap, alpha=args.alpha).save(out)
    print(f"top prediction: {predictor.class_names[class_idx]} (class {class_idx})")
    print(f"saved Grad-CAM overlay to {out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .inference.demo import run_demo

    run_demo(args.checkpoint, device=args.device, share=args.share)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Print params (and FLOPs when ptflops is available) for all registry models."""
    from .models.registry import available_models, create_model, get_num_parameters

    print(f"{'model':32s} {'params':>10s} {'MACs':>12s}")
    print("-" * 58)
    for name in available_models():
        try:
            model = create_model(name, num_classes=6, pretrained=False).eval()
            params = get_num_parameters(model) / 1e6
            macs = "n/a"
            try:
                from ptflops import get_model_complexity_info

                macs, _ = get_model_complexity_info(
                    model, (3, args.input_size, args.input_size), as_strings=True, print_per_layer_stat=False
                )
            except Exception:
                pass
            print(f"{name:32s} {params:8.2f}M {macs:>12s}")
        except Exception as e:
            print(f"{name:32s} FAILED: {e}")
    return 0


# ---- entry point ------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="garbage", description=__doc__)
    parser.add_argument("--version", action="version", version=f"garbage-classifier {__version__}")
    _add_debug_arg(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("show-config", help="print the fully resolved YAML configuration")
    _add_debug_arg(p)
    _add_config_args(p)
    p.set_defaults(func=cmd_show_config)

    p = sub.add_parser("prepare-data", help="generate manifests from class folders")
    _add_debug_arg(p)
    p.add_argument("--data-dir", type=str, default=None, help="dir with one subfolder per class")
    p.add_argument("--strict", action="store_true", help="fail if duplicate images are found")
    _add_config_args(p)
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("train", help="train a model")
    _add_debug_arg(p)
    _add_config_args(p)
    p.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    p.add_argument("--dry-run", action="store_true", help="train on 1 batch only to verify the pipeline")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    _add_debug_arg(p)
    _add_config_args(p)
    p.add_argument("--checkpoint", type=str, required=True, help="checkpoint .pt path")
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--error-limit", type=int, default=20)
    p.add_argument("--plot", action="store_true", help="save a confusion-matrix PNG next to the checkpoint")
    p.add_argument("--tta", action="store_true", help="test-time augmentation (average over horizontal flip)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="predict image(s) from a checkpoint")
    _add_debug_arg(p)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True, help="image path or directory")
    p.add_argument("--top-k", type=_positive_int, default=3)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tta", action="store_true", help="test-time augmentation (average over horizontal flip)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("export-onnx", help="export a checkpoint to ONNX")
    _add_debug_arg(p)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="model.onnx", help="output .onnx path")
    p.add_argument("--image-size", type=int, default=None, help="input size (default: from checkpoint config)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", type=str, default="cpu", help="device for export (default: cpu)")
    p.add_argument("--no-verify", action="store_true", help="skip onnxruntime sanity check")
    p.set_defaults(func=cmd_export_onnx)

    p = sub.add_parser("explain", help="Grad-CAM heatmap for a single image")
    _add_debug_arg(p)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True, help="image path")
    p.add_argument("--output", type=str, default="gradcam.png")
    p.add_argument("--class-idx", type=_nonnegative_int, default=None, help="class of interest (default: top-1)")
    p.add_argument("--alpha", type=_unit_float, default=0.5, help="heatmap blend strength")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("bench", help="list registry models with params/FLOPs")
    _add_debug_arg(p)
    p.add_argument("--input-size", type=int, default=224)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("demo", help="launch the Gradio web demo")
    _add_debug_arg(p)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--share", action="store_true", help="create a public share link")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ManifestError, ValueError, KeyError) as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
