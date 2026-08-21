"""Command-line interface for reproducible image classification workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from . import __version__
from .config import ExperimentConfig, load_config, load_config_with_sources, to_dict
from .data import inspect_prepared_data, verify_prepared_data
from .data.manifest import ManifestError
from .data.prepare import prepare_data
from .evaluation.evaluate import evaluate_checkpoint
from .preflight import PreflightError
from .training.checkpoint import CheckpointCompatibilityError
from .training.train import train_from_config
from .utils import setup_logging, write_text_atomic


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="YAML configuration file")
    parser.add_argument(
        "--set",
        nargs=2,
        action="append",
        default=[],
        metavar=("KEY", "VALUE"),
        help="override a dotted key, for example --set train.lr 1e-4",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--output-dir", type=str, default=None, help="output root or command output directory")


def _parse_set(overrides: list[list[str]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in overrides:
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


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
        "--debug", action="store_true", default=argparse.SUPPRESS, help="show tracebacks for command errors"
    )


def _resolve_cfg(args: argparse.Namespace) -> ExperimentConfig:
    cfg = load_config(args.config, overrides=_parse_set(args.set))
    if args.device != "auto":
        cfg = replace(cfg, device=args.device)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=Path(args.output_dir))
    return cfg


def cmd_show_config(args: argparse.Namespace) -> int:
    import yaml

    if not getattr(args, "sources", False):
        print(yaml.safe_dump(to_dict(_resolve_cfg(args)), sort_keys=False), end="")
        return 0
    cfg, sources = load_config_with_sources(args.config, _parse_set(args.set))
    if args.device != "auto":
        cfg = replace(cfg, device=args.device)
        sources["device"] = "cli"
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=Path(args.output_dir))
        sources["output_dir"] = "cli"
    print(yaml.safe_dump({"config": to_dict(cfg), "sources": dict(sorted(sources.items()))}, sort_keys=False), end="")
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    prepare_data(
        data_dir=args.data_dir or cfg.data.data_dir,
        manifest_dir=cfg.data.manifest_dir,
        split_ratios=cfg.data.split_ratios,
        seed=cfg.data.seed,
        strict=args.strict,
        overwrite=getattr(args, "overwrite", False),
    )
    return 0


def cmd_verify_data(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    metadata = verify_prepared_data(cfg.data.manifest_dir, args.data_dir or cfg.data.data_dir)
    print(f"verified dataset {metadata.identity} ({sum(metadata.split_counts.values())} images)")
    return 0


def cmd_inspect_data(args: argparse.Namespace) -> int:
    import yaml

    cfg = _resolve_cfg(args)
    report = inspect_prepared_data(cfg.data.manifest_dir, args.data_dir or cfg.data.data_dir)
    print(yaml.safe_dump(report, sort_keys=False), end="")
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
        overwrite=getattr(args, "overwrite", False),
        config_path=args.config,
    )
    return 0


def cmd_compare_runs(args: argparse.Namespace) -> int:
    from .evaluation.comparison import compare_runs, write_comparison

    rows = compare_runs(args.run_dirs, metric=args.metric)
    if args.output is not None:
        write_comparison(args.output, rows)
    print(f"{'rank':>4s} {'run':24s} {'epoch':>5s} {'metric':18s} {'value':>9s}")
    for rank, row in enumerate(rows, start=1):
        print(f"{rank:4d} {row.run_name:24s} {row.epoch:5d} {row.metric:18s} {row.metric_value:9.4f}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .inference import Predictor

    target = Path(args.image)
    suffixes = {".jpg", ".jpeg", ".png"}
    if not target.exists():
        raise FileNotFoundError(f"input path does not exist: {target}")
    if target.is_file():
        if target.suffix.lower() not in suffixes:
            raise ValueError(f"unsupported image file: {target}")
        targets = [target]
    elif target.is_dir():
        iterator = target.rglob("*") if getattr(args, "recursive", False) else target.iterdir()
        targets = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in suffixes)
        if not targets:
            raise ValueError(f"no supported images found in directory: {target}")
    else:
        raise ValueError(f"input path is not a file or directory: {target}")

    predictor = Predictor(args.checkpoint, device=args.device, config_path=args.config)
    records = []
    for path in targets:
        top = predictor.predict_path(path, top_k=args.top_k, tta=args.tta)
        record = {
            "path": str(path),
            "predictions": [{"class_name": name, "probability": probability} for name, probability in top],
        }
        records.append(record)
        print(f"{path} -> " + ", ".join(f"{name}={probability:.3f}" for name, probability in top))
    if getattr(args, "output", None) is not None:
        output = Path(args.output)
        if output.exists() and not getattr(args, "overwrite", False):
            raise FileExistsError(f"prediction output already exists: {output}; use --overwrite")
        write_text_atomic(
            output,
            json.dumps({"checkpoint": str(args.checkpoint), "predictions": records}, indent=2, ensure_ascii=False)
            + "\n",
        )
    return 0


def cmd_export_onnx(args: argparse.Namespace) -> int:
    from .inference.export import export_onnx

    output = export_onnx(
        args.checkpoint,
        args.output,
        image_size=args.image_size,
        opset=args.opset,
        verify=not args.no_verify,
        device=args.device,
        overwrite=getattr(args, "overwrite", False),
        config_path=args.config,
    )
    print(f"exported ONNX model to {output}")
    print(f"metadata sidecar: {output.with_suffix('.onnx.meta.yaml')}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from .inference import Predictor
    from .inference.gradcam import GradCAM, overlay_heatmap
    from .models.registry import model_spec

    predictor = Predictor(args.checkpoint, device=args.device, config_path=args.config)
    if args.class_idx is not None and args.class_idx >= len(predictor.class_names):
        raise ValueError(
            f"class index {args.class_idx} is outside the valid range [0, {len(predictor.class_names) - 1}]"
        )
    spec = model_spec(predictor.cfg.model.name)
    if not spec.supports_gradcam:
        raise ValueError(f"Grad-CAM is not supported for {spec.name}; choose a CNN model or a model-specific explainer")
    with Image.open(args.image) as image:
        tensor = predictor.transform(image.convert("RGB")).to(predictor.device)
        cam = GradCAM(predictor.model)
        try:
            heatmap, class_idx = cam.generate(tensor, class_idx=args.class_idx, device=predictor.device)
        finally:
            cam.close()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        overlay_heatmap(image, heatmap, alpha=args.alpha).save(args.output)
    print(f"top prediction: {predictor.class_names[class_idx]} (class {class_idx})")
    print(f"saved Grad-CAM overlay to {args.output}")
    return 0


def cmd_list_models(_args: argparse.Namespace) -> int:
    from .models.registry import available_model_specs

    print(f"{'model':32s} {'provider':12s} {'input':>5s} {'Grad-CAM':>8s}")
    print("-" * 66)
    for spec in available_model_specs():
        print(f"{spec.name:32s} {spec.provider:12s} {spec.input_size:>5d} {str(spec.supports_gradcam):>8s}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Backward-compatible alias for metadata-only model listing."""
    return cmd_list_models(args)


def cmd_demo(args: argparse.Namespace) -> int:
    from .inference.demo import run_demo

    run_demo(args.checkpoint, device=args.device, share=args.share, config_path=args.config)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="garbage", description=__doc__)
    parser.add_argument("--version", action="version", version=f"garbage-classifier {__version__}")
    _add_debug_arg(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show-config", help="print the fully resolved configuration")
    _add_debug_arg(show)
    _add_config_args(show)
    show.add_argument("--sources", action="store_true", help="show the origin of every resolved field")
    show.set_defaults(func=cmd_show_config)

    prepare = subparsers.add_parser("prepare-data", help="generate audited train/valid/test manifests")
    _add_debug_arg(prepare)
    prepare.add_argument("--data-dir", type=str, default=None, help="class-folder source directory")
    prepare.add_argument("--strict", action="store_true", help="fail when duplicate images are found")
    prepare.add_argument("--overwrite", action="store_true", help="replace an existing manifest directory")
    _add_config_args(prepare)
    prepare.set_defaults(func=cmd_prepare_data)

    data_commands: tuple[tuple[str, Callable[[argparse.Namespace], int], str], ...] = (
        ("verify-data", cmd_verify_data, "verify prepared manifests and source image identity"),
        ("inspect-data", cmd_inspect_data, "print prepared data identity and class distribution"),
    )
    for command, handler, help_text in data_commands:
        data_command = subparsers.add_parser(command, help=help_text)
        _add_debug_arg(data_command)
        data_command.add_argument("--data-dir", type=str, default=None)
        _add_config_args(data_command)
        data_command.set_defaults(func=handler)

    train = subparsers.add_parser("train", help="train a verified, isolated experiment")
    _add_debug_arg(train)
    _add_config_args(train)
    train.add_argument("--resume", type=str, default=None, help="last.pt in the same run directory")
    train.add_argument("--dry-run", action="store_true", help="run one forward/backward batch without writing a run")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint and publish evidence")
    _add_debug_arg(evaluate)
    _add_config_args(evaluate)
    evaluate.add_argument("--checkpoint", type=str, required=True, help="checkpoint .pt path")
    evaluate.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    evaluate.add_argument("--error-limit", type=_nonnegative_int, default=20)
    evaluate.add_argument(
        "--plot", action="store_true", help="write confusion and reliability plots (requires matplotlib)"
    )
    evaluate.add_argument("--tta", action="store_true", help="average original and horizontally flipped probabilities")
    evaluate.add_argument("--overwrite", action="store_true", help="replace an existing evaluation directory")
    evaluate.set_defaults(func=cmd_evaluate)

    compare = subparsers.add_parser("compare-runs", help="rank compatible completed runs by one validation metric")
    _add_debug_arg(compare)
    compare.add_argument("run_dirs", nargs="+", help="two or more completed run directories")
    compare.add_argument("--metric", type=str, default="macro_f1")
    compare.add_argument("--output", type=str, default=None, help="optional new comparison CSV path")
    compare.set_defaults(func=cmd_compare_runs)

    predict = subparsers.add_parser("predict", help="predict one image or a directory")
    _add_debug_arg(predict)
    predict.add_argument("--checkpoint", type=str, required=True)
    predict.add_argument("--image", type=str, required=True, help="image path or directory")
    predict.add_argument("--top-k", type=_positive_int, default=3)
    predict.add_argument("--config", type=str, default=None)
    predict.add_argument("--device", type=str, default="auto")
    predict.add_argument("--tta", action="store_true")
    predict.add_argument("--recursive", action="store_true", help="recurse through an input directory")
    predict.add_argument("--output", type=str, default=None, help="optional JSON evidence path")
    predict.add_argument("--overwrite", action="store_true", help="replace an existing JSON output")
    predict.set_defaults(func=cmd_predict)

    export = subparsers.add_parser("export-onnx", help="export a checkpoint to ONNX")
    _add_debug_arg(export)
    export.add_argument("--checkpoint", type=str, required=True)
    export.add_argument("--output", type=str, default="model.onnx")
    export.add_argument("--image-size", type=_positive_int, default=None)
    export.add_argument("--opset", type=_positive_int, default=17)
    export.add_argument("--device", type=str, default="cpu")
    export.add_argument("--config", type=str, default=None, help="reviewed config required for external factories")
    export.add_argument("--no-verify", action="store_true")
    export.add_argument("--overwrite", action="store_true", help="replace an existing ONNX output")
    export.set_defaults(func=cmd_export_onnx)

    explain = subparsers.add_parser("explain", help="write a Grad-CAM overlay for one CNN prediction")
    _add_debug_arg(explain)
    explain.add_argument("--checkpoint", type=str, required=True)
    explain.add_argument("--image", type=str, required=True)
    explain.add_argument("--output", type=str, default="gradcam.png")
    explain.add_argument("--class-idx", type=_nonnegative_int, default=None)
    explain.add_argument("--alpha", type=_unit_float, default=0.5)
    explain.add_argument("--config", type=str, default=None)
    explain.add_argument("--device", type=str, default="auto")
    explain.set_defaults(func=cmd_explain)

    model_commands: tuple[tuple[str, Callable[[argparse.Namespace], int]], ...] = (
        ("list-models", cmd_list_models),
        ("bench", cmd_bench),
    )
    for command, handler in model_commands:
        list_models = subparsers.add_parser(command, help="list model specifications without constructing models")
        _add_debug_arg(list_models)
        list_models.add_argument(
            "--input-size",
            type=_positive_int,
            default=None,
            help="legacy compatibility option; listing uses model specs",
        )
        list_models.set_defaults(func=handler)

    demo = subparsers.add_parser("demo", help="launch the optional Gradio demo")
    _add_debug_arg(demo)
    demo.add_argument("--checkpoint", type=str, required=True)
    demo.add_argument("--device", type=str, default="auto")
    demo.add_argument("--config", type=str, default=None, help="reviewed config required for external factories")
    demo.add_argument("--share", action="store_true")
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        CheckpointCompatibilityError,
        FileExistsError,
        FileNotFoundError,
        ManifestError,
        PreflightError,
        ValueError,
        KeyError,
    ) as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
