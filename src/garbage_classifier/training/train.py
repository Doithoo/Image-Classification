"""Training command logic: build datasets/loaders and run the Trainer."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..config import ExperimentConfig, dump_config
from ..data import ImageClassificationDataset, collate_fn, manifest_classes
from ..data.transforms import build_eval_transform, build_train_transform
from ..models.registry import create_model, get_num_parameters
from ..utils import git_revision, pick_device, set_all_seeds
from .trainer import Trainer
from .weights import build_weighted_sampler, compute_class_weights

logger = logging.getLogger("garbage_classifier.train")


def train_from_config(
    cfg: ExperimentConfig,
    resume: str | None = None,
    dry_run: bool = False,
    run_name: str | None = None,
) -> Path:
    """Run a full training experiment from config; returns the run directory.

    ``dry_run`` trains on a single batch to verify the pipeline end-to-end
    (data → model → forward → backward) before committing to a long run.
    """
    set_all_seeds(cfg.train.seed)
    device = pick_device(cfg.device)
    logger.info("garbage-classifier | git %s | device %s", git_revision(), device)

    run_dir = Path(cfg.output_dir) / (
        run_name or cfg.run_name or f"{cfg.model.name}-{__import__('time').strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    logger.info("run dir: %s", run_dir)

    class_names = manifest_classes(cfg.data.manifest_dir)
    if len(class_names) != cfg.model.num_classes:
        raise ValueError(
            f"model.num_classes={cfg.model.num_classes} does not match manifest class count {len(class_names)}"
        )

    train_ds = ImageClassificationDataset(
        Path(cfg.data.manifest_dir) / "train.csv", transform=build_train_transform(cfg.data)
    )
    valid_ds = ImageClassificationDataset(
        Path(cfg.data.manifest_dir) / "valid.csv", transform=build_eval_transform(cfg.data)
    )
    if len(train_ds) == 0:
        raise ValueError("training dataset is empty; add at least one sample to train.csv")
    if len(valid_ds) == 0:
        raise ValueError("validation dataset is empty; add at least one sample to valid.csv")

    # class-imbalance handling: loss weights and/or weighted sampling (ablation support)
    train_counts = [0] * len(class_names)
    for _, label in train_ds.samples:
        train_counts[label] += 1
    class_weights = compute_class_weights(train_counts, cfg.train.class_weight)
    if class_weights:
        logger.info("class weights (%s): %s", cfg.train.class_weight, [round(w, 3) for w in class_weights])
    sampler = None
    if cfg.train.sampler == "weighted":
        sampler = build_weighted_sampler([label for _, label in train_ds.samples], train_counts)
        logger.info("using WeightedRandomSampler (rare classes oversampled)")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        collate_fn=collate_fn,
        # Only discard a singleton tail, which can make BatchNorm fail. Tiny
        # datasets must still yield their one partial batch.
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

    model = create_model(cfg.model.name, num_classes=len(class_names), pretrained=cfg.model.pretrained)
    logger.info("model=%s params=%.2fM", cfg.model.name, get_num_parameters(model) / 1e6)

    if dry_run:
        # 1-batch sanity check: verifies data pipeline + forward/backward before a long run
        logger.info("dry-run: training on 1 batch only ...")
        model = model.to(device)
        model.train()
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=cfg.train.amp and device.type in ("cuda", "mps")
        ):
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        logger.info("dry-run OK: input=%s output=%s loss=%.3f", tuple(images.shape), tuple(outputs.shape), loss.item())
        return run_dir

    trainer = Trainer(model, cfg, device, class_names, run_dir, class_weights=class_weights)
    result = trainer.fit(train_loader, valid_loader, resume_from=resume)
    logger.info(
        "done: epochs=%d best_%s=%.4f elapsed=%.1fmin  (best.pt in %s)",
        result["epochs_run"],
        result["best_metric_name"],
        result["best_metric"],
        result["elapsed_min"],
        run_dir,
    )
    return run_dir
