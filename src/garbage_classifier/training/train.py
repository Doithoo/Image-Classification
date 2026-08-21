"""Training orchestration: verified data, preflight, model construction and run publication."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..config import ExperimentConfig, dump_config
from ..data import ImageClassificationDataset, collate_fn, verify_prepared_data
from ..data.transforms import build_eval_transform, build_train_transform
from ..models.registry import create_model, get_num_parameters, resolve_preprocessing
from ..preflight import validate_training_request
from ..utils import git_revision, pick_device, set_all_seeds
from .metadata import build_run_metadata, write_run_metadata
from .trainer import Trainer
from .weights import build_weighted_sampler, compute_class_weights

logger = logging.getLogger("garbage_classifier.train")


def _resolve_run_dir(cfg: ExperimentConfig, run_name: str | None) -> Path:
    name = run_name or cfg.run_name or f"{cfg.model.name}-{__import__('time').strftime('%Y%m%d-%H%M%S')}"
    return cfg.output_dir / name


def _validate_run_destination(run_dir: Path, resume: str | None, *, dry_run: bool) -> None:
    if dry_run:
        if resume is not None:
            raise ValueError("--resume cannot be combined with --dry-run")
        return
    if resume is None:
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}; choose a new run_name or use --resume")
        return
    checkpoint = Path(resume).resolve()
    if checkpoint.name != "last.pt":
        raise ValueError("resume requires last.pt; best.pt contains deployable weights, not the latest training state")
    if checkpoint.parent != run_dir.resolve():
        raise ValueError("resume checkpoint must be inside the resolved run directory")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")


def train_from_config(
    cfg: ExperimentConfig,
    resume: str | None = None,
    dry_run: bool = False,
    run_name: str | None = None,
) -> Path:
    """Train one isolated experiment after validating data and output contracts."""
    metadata = verify_prepared_data(cfg.data.manifest_dir, cfg.data.data_dir)
    run_dir = _resolve_run_dir(cfg, run_name)
    report = validate_training_request(cfg, metadata, run_dir)
    report.raise_for_issues()
    _validate_run_destination(run_dir, resume, dry_run=dry_run)
    for notice in report.notices:
        logger.warning("preflight: %s", notice)

    class_names = list(metadata.classes)
    if cfg.model.num_classes is not None and cfg.model.num_classes != len(class_names):
        raise ValueError(
            f"model.num_classes={cfg.model.num_classes} does not match manifest class count {len(class_names)}"
        )
    cfg = replace(
        cfg,
        data=resolve_preprocessing(cfg.data, cfg.model),
        model=replace(cfg.model, num_classes=len(class_names)),
    )
    set_all_seeds(cfg.train.seed)
    device = pick_device(cfg.device)
    logger.info("garbage-classifier | git %s | device %s | dataset %s", git_revision(), device, metadata.identity[:12])

    # A dry run intentionally leaves no run artifacts behind.

    train_ds = ImageClassificationDataset(
        cfg.data.manifest_dir / "train.csv",
        root_dir=cfg.data.data_dir,
        transform=build_train_transform(cfg.data),
    )
    valid_ds = ImageClassificationDataset(
        cfg.data.manifest_dir / "valid.csv",
        root_dir=cfg.data.data_dir,
        transform=build_eval_transform(cfg.data),
    )
    if len(train_ds) == 0:
        raise ValueError("training dataset is empty; add at least one sample to train.csv")
    if len(valid_ds) == 0:
        raise ValueError("validation dataset is empty; add at least one sample to valid.csv")

    train_counts = [0] * len(class_names)
    for _path, label in train_ds.samples:
        train_counts[label] += 1
    class_weights = compute_class_weights(train_counts, cfg.train.class_weight)
    sampler = (
        build_weighted_sampler([label for _path, label in train_ds.samples], train_counts)
        if cfg.train.sampler == "weighted"
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        collate_fn=collate_fn,
        drop_last=len(train_ds) >= cfg.train.batch_size and len(train_ds) % cfg.train.batch_size == 1,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        collate_fn=collate_fn,
    )
    logger.info("train=%d valid=%d classes=%s", len(train_ds), len(valid_ds), class_names)

    model = create_model(
        cfg.model.name,
        num_classes=len(class_names),
        pretrained=cfg.model.pretrained,
        factory=cfg.model.factory,
        params=cfg.model.params,
    )
    logger.info("model=%s params=%.2fM", cfg.model.name, get_num_parameters(model) / 1e6)
    if dry_run:
        model = model.to(device).train()
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=cfg.train.amp and device.type in {"cuda", "mps"}
        ):
            loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        logger.info("dry-run OK: input=%s loss=%.3f", tuple(images.shape), loss.item())
        return run_dir

    if resume is None:
        run_dir.mkdir(parents=True)
        dump_config(cfg, run_dir / "config.yaml")
    elif not (run_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"resumed run is missing resolved config: {run_dir / 'config.yaml'}")
    started_at = datetime.now(timezone.utc)
    trainer = Trainer(model, cfg, device, class_names, run_dir, metadata.identity, class_weights=class_weights)
    result = trainer.fit(train_loader, valid_loader, resume_from=resume)
    finished_at = datetime.now(timezone.utc)
    run_metadata = build_run_metadata(
        device,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=result["elapsed_min"] * 60,
        manifest_identity=metadata.identity,
        dataset_schema_version=metadata.schema_version,
    )
    write_run_metadata(run_dir / "run.yaml", run_metadata)
    logger.info(
        "done: epochs=%d best_%s=%.4f elapsed=%.1fmin  (best.pt in %s)",
        result["epochs_run"],
        result["best_metric_name"],
        result["best_metric"],
        result["elapsed_min"],
        run_dir,
    )
    return run_dir
