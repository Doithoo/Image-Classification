"""Command-line interface.

Commands (config overrides via dotted keys, e.g. ``--set train.lr 1e-4``):
    prepare-data   Generate portable train/valid/test manifests from class folders
    train          Train a model (config-driven, resumable, AMP, early stopping)
    evaluate       Evaluate a checkpoint on the test split (full metrics)
    predict        Predict a single image or a folder of images
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from . import __version__
from .config import ExperimentConfig, dump_config, load_config
from .data import ImageClassificationDataset, build_manifest, collate_fn, manifest_classes
from .data.transforms import build_eval_transform, build_train_transform
from .evaluation import classification_report, error_samples
from .inference import Predictor
from .models.registry import create_model, get_num_parameters
from .training import Trainer
from .utils import git_revision, pick_device, setup_logging


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


def _resolve_cfg(args: argparse.Namespace) -> ExperimentConfig:
    cfg = load_config(args.config, overrides=_parse_set(args.set))
    if args.device != "auto":
        cfg.device = args.device
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    return cfg


def _run_dir(cfg: ExperimentConfig, model_name: str) -> Path:
    run_name = cfg.run_name or f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    return Path(cfg.output_dir) / run_name


# ---- prepare-data ---------------------------------------------------------
def cmd_prepare_data(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    data_dir = args.data_dir or cfg.data.data_dir
    from .data.manifest import find_duplicates

    dups = find_duplicates(data_dir)
    if dups:
        n = sum(len(g) - 1 for g in dups)
        print(f"warning: {n} duplicate images found (same content, different names); e.g. {dups[0][:2]}")
        if args.strict:
            print("strict mode: aborting")
            return 1
    manifests = build_manifest(
        data_dir,
        cfg.data.manifest_dir,
        split_ratios=cfg.data.split_ratios,
        seed=cfg.data.seed,
        validate=True,
    )
    print(f"manifests written to {cfg.data.manifest_dir}:")
    for split, path in manifests.items():
        print(f"  {split:6s} {path}")
    print(f"summary: {Path(cfg.data.manifest_dir) / 'summary.txt'}")
    return 0


# ---- train ----------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    log = setup_logging(cfg.log_level)
    from .utils import set_all_seeds

    set_all_seeds(cfg.train.seed)
    device = pick_device(cfg.device)
    log.info("garbage-classifier %s | git %s | device %s", __version__, git_revision(), device)

    run_dir = _run_dir(cfg, cfg.model.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    log.info("run dir: %s", run_dir)

    try:
        class_names = manifest_classes(cfg.data.manifest_dir)
        if len(class_names) != cfg.model.num_classes:
            log.warning(
                "manifest classes (%d) differ from config num_classes (%d); using manifest",
                len(class_names),
                cfg.model.num_classes,
            )
    except Exception:
        class_names = cfg.data.classes
    train_ds = ImageClassificationDataset(
        Path(cfg.data.manifest_dir) / "train.csv", transform=build_train_transform(cfg.data)
    )
    valid_ds = ImageClassificationDataset(
        Path(cfg.data.manifest_dir) / "valid.csv", transform=build_eval_transform(cfg.data)
    )

    # class-imbalance handling: loss weights and/or weighted sampling (ablation support)
    from .training.weights import build_weighted_sampler, compute_class_weights

    train_counts = [0] * len(class_names)
    for _, label in train_ds.samples:
        train_counts[label] += 1
    class_weights = compute_class_weights(train_counts, cfg.train.class_weight)
    if class_weights:
        log.info("class weights (%s): %s", cfg.train.class_weight, [round(w, 3) for w in class_weights])
    sampler = None
    if cfg.train.sampler == "weighted":
        sampler = build_weighted_sampler([label for _, label in train_ds.samples], train_counts)
        log.info("using WeightedRandomSampler (rare classes oversampled)")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        collate_fn=collate_fn,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        collate_fn=collate_fn,
    )
    log.info("train=%d valid=%d classes=%s", len(train_ds), len(valid_ds), class_names)

    model = create_model(cfg.model.name, num_classes=len(class_names), pretrained=cfg.model.pretrained)
    log.info("model=%s params=%.2fM", cfg.model.name, get_num_parameters(model) / 1e6)

    resume = args.resume
    trainer = Trainer(model, cfg, device, class_names, run_dir, class_weights=class_weights)
    result = trainer.fit(train_loader, valid_loader, resume_from=resume)
    log.info(
        "done: epochs=%d best_%s=%.4f elapsed=%.1fmin  (best.pt in %s)",
        result["epochs_run"],
        result["best_metric_name"],
        result["best_metric"],
        result["elapsed_min"],
        run_dir,
    )
    return 0


# ---- evaluate -------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    log = setup_logging(cfg.log_level)
    device = pick_device(cfg.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names: list[str] = payload["class_names"]
    # model identity comes from the checkpoint metadata (self-contained); explicit
    # --config may override data paths but not silently swap the architecture
    ckpt_model = payload.get("config", {}).get("model", {}).get("name", cfg.model.name)
    log.info("evaluating %s (%s) on %s (classes=%d)", args.checkpoint, ckpt_model, args.split, len(class_names))

    test_ds = ImageClassificationDataset(
        Path(cfg.data.manifest_dir) / f"{args.split}.csv", transform=build_eval_transform(cfg.data)
    )
    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        collate_fn=collate_fn,
    )

    from .inference.predictor import Predictor

    predictor = Predictor(args.checkpoint, device=cfg.device, config_path=args.config)
    predictor.model.load_state_dict(payload["model_state_dict"])

    all_preds, all_labels, all_paths = [], [], []
    sample_paths = [p for p, _ in test_ds.samples]
    start = 0
    with torch.no_grad():
        for images, labels in loader:
            probs = predictor.predict_probs(images.to(device), tta=args.tta)
            preds_batch = probs.argmax(dim=1).cpu()
            all_preds.append(preds_batch)
            all_labels.append(labels)
            end = start + len(labels)
            all_paths.extend(sample_paths[start:end])
            start = end

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    from .evaluation import evaluate_predictions

    metrics = evaluate_predictions(preds, labels, num_classes=len(class_names))
    print(classification_report(metrics, class_names))
    print(f"\nconfusion matrix (rows=true, cols=pred):\n{metrics['confusion']}")

    # prediction CSV + error list
    out_dir = Path(args.output_dir or Path(args.checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt
        import numpy as np

        cm = np.array(metrics["confusion"], dtype=np.int64)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix (acc={metrics['accuracy']:.3f})")
        thresh = cm.max() / 2
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        print(f"confusion matrix plot saved to {out_dir / 'confusion_matrix.png'}")

    # prediction CSV + error list
    out_dir = Path(args.output_dir or Path(args.checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w") as f:
        f.write("path,true,pred\n")
        for p, t, pr in zip(all_paths, labels, preds, strict=True):
            f.write(f"{p},{class_names[t]},{class_names[pr]}\n")
    err = error_samples(labels, preds, all_paths, limit=args.error_limit)
    err_csv = out_dir / "errors.csv"
    with err_csv.open("w") as f:
        f.write("path,true,pred\n")
        for e in err:
            f.write(f"{e['path']},{class_names[e['true']]},{class_names[e['pred']]}\n")
    log.info("wrote %s and %s", pred_csv, err_csv)
    return 0


# ---- predict --------------------------------------------------------------
def cmd_predict(args: argparse.Namespace) -> int:
    predictor = Predictor(args.checkpoint, device=args.device, config_path=args.config)
    target = Path(args.image)
    targets = [target] if target.is_file() else sorted(target.glob("*"))
    for path in targets:
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        top = predictor.predict_path(path, top_k=args.top_k, tta=args.tta)
        ranked = ", ".join(f"{name}={prob:.3f}" for name, prob in top)
        print(f"{path} -> {ranked}")
    return 0


# ---- export-onnx -----------------------------------------------------------
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


# ---- explain (Grad-CAM) ------------------------------------------------------
def cmd_explain(args: argparse.Namespace) -> int:
    from .inference.gradcam import GradCAM, overlay_heatmap
    from .inference.predictor import Predictor

    predictor = Predictor(args.checkpoint, device=args.device, config_path=args.config)
    image = Image.open(args.image)
    img_tensor = predictor.transform(image.convert("RGB")).to(predictor.device)

    cam_model = GradCAM(predictor.model)
    heatmap, class_idx = cam_model.generate(img_tensor, class_idx=args.class_idx, device=predictor.device)

    out = Path(args.output)
    overlay_heatmap(image, heatmap, alpha=args.alpha).save(out)
    print(f"top prediction: {predictor.class_names[class_idx]} (class {class_idx})")
    print(f"saved Grad-CAM overlay to {out}")
    return 0


# ---- demo -------------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> int:
    from .inference.demo import run_demo

    run_demo(args.checkpoint, device=args.device, share=args.share)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="garbage", description=__doc__)
    parser.add_argument("--version", action="version", version=f"garbage-classifier {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data", help="generate manifests from class folders")
    p.add_argument("--data-dir", type=str, default=None, help="dir with one subfolder per class")
    p.add_argument("--strict", action="store_true", help="fail if duplicate images are found")
    _add_config_args(p)
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("train", help="train a model")
    _add_config_args(p)
    p.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    _add_config_args(p)
    p.add_argument("--checkpoint", type=str, required=True, help="checkpoint .pt path")
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--error-limit", type=int, default=20)
    p.add_argument("--plot", action="store_true", help="save a confusion-matrix PNG next to the checkpoint")
    p.add_argument("--tta", action="store_true", help="test-time augmentation (average over horizontal flip)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="predict image(s) from a checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True, help="image path or directory")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tta", action="store_true", help="test-time augmentation (average over horizontal flip)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("export-onnx", help="export a checkpoint to ONNX")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="model.onnx", help="output .onnx path")
    p.add_argument("--image-size", type=int, default=None, help="input size (default: from checkpoint config)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", type=str, default="cpu", help="device for export (default: cpu)")
    p.add_argument("--no-verify", action="store_true", help="skip onnxruntime sanity check")
    p.set_defaults(func=cmd_export_onnx)

    p = sub.add_parser("explain", help="Grad-CAM heatmap for a single image")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True, help="image path")
    p.add_argument("--output", type=str, default="gradcam.png")
    p.add_argument("--class-idx", type=int, default=None, help="class of interest (default: top-1)")
    p.add_argument("--alpha", type=float, default=0.5, help="heatmap blend strength")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("demo", help="launch the Gradio web demo")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--share", action="store_true", help="create a public share link")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
