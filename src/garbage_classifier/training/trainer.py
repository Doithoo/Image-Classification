"""Trainer: training loop with AMP, resume, early stopping and best-checkpointing."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..evaluation.metrics import evaluate_predictions
from ..utils import CsvLogger
from .checkpoint import load_checkpoint, save_checkpoint

logger = logging.getLogger("garbage_classifier.trainer")


class Trainer:
    """Wraps the full train/validate loop.

    Highlights (vs the legacy train.py):
    - AMP (autocast + GradScaler) on cuda/mps
    - best/last checkpoints with full metadata (resume-safe)
    - early stopping on the configured metric
    - per-epoch CSV metrics + best-metric tracking
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        device: torch.device,
        class_names: list[str],
        output_dir: str | Path,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        t = cfg.train
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)
        params = [p for p in model.parameters() if p.requires_grad]
        if t.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=t.lr, momentum=t.momentum, weight_decay=t.weight_decay)
        elif t.optimizer == "lion":
            from lion_pytorch import Lion

            self.optimizer = Lion(params, lr=t.lr, weight_decay=t.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)

        self.scheduler: Any | None = None
        if t.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t.epochs)
        elif t.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, t.epochs // 3), gamma=0.1)

        self.scaler = torch.amp.GradScaler("cuda", enabled=(t.amp and device.type == "cuda"))
        self.use_amp = t.amp and device.type in ("cuda", "mps")

        self.epoch = 0
        self.best_metric = -float("inf")
        self.patience_left = t.early_stop_patience

    # ---- public API -------------------------------------------------------
    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, resume_from: str | None = None) -> dict[str, Any]:
        csv_path = self.output_dir / "metrics.csv"
        csv_logger = CsvLogger(csv_path, ["epoch", "train_loss", "val_loss", "accuracy", "balanced_acc", "macro_f1"])

        if resume_from is not None:
            self._resume(resume_from)

        start = time.time()
        for epoch in range(self.epoch, self.cfg.train.epochs):
            train_loss = self._run_epoch(train_loader, train=True)["loss"]
            metrics = self._run_epoch(valid_loader, train=False)

            lr = self.optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": round(metrics["loss"], 4),
                "accuracy": round(metrics["accuracy"], 4),
                "balanced_acc": round(metrics["balanced_accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
            }
            csv_logger.write(row)
            logger.info(
                "epoch %d/%d lr=%.2e train_loss=%.4f val_loss=%.4f acc=%.4f bal_acc=%.4f macro_f1=%.4f",
                epoch + 1,
                self.cfg.train.epochs,
                lr,
                train_loss,
                metrics["loss"],
                metrics["accuracy"],
                metrics["balanced_accuracy"],
                metrics["macro_f1"],
            )

            self.epoch = epoch + 1
            score = metrics.get(self.cfg.train.best_metric, metrics["accuracy"])
            improved = score > self.best_metric + 1e-6
            if improved:
                self.best_metric = score
                self.patience_left = self.cfg.train.early_stop_patience
                self._save("best.pt", metrics)
            else:
                self.patience_left -= 1
            self._save("last.pt", metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.patience_left <= 0:
                logger.info("early stopping after %d epochs without improvement", epoch + 1)
                break

        return {
            "epochs_run": self.epoch,
            "best_metric": self.best_metric,
            "best_metric_name": self.cfg.train.best_metric,
            "elapsed_min": (time.time() - start) / 60,
        }

    # ---- internals --------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        n_batches = 0

        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                if train:
                    if self.use_amp and self.device.type == "cuda":
                        self.scaler.scale(loss).backward()
                        if self.cfg.train.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.cfg.train.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                        self.optimizer.step()

                if not train:
                    all_preds.append(outputs.argmax(dim=1).detach().cpu())
                    all_labels.append(labels.detach().cpu())
                total_loss += loss.item()
                n_batches += 1

        if train:
            return {"loss": total_loss / max(n_batches, 1)}
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        metrics = evaluate_predictions(preds, labels, num_classes=self.cfg.model.num_classes)
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics

    def _save(self, name: str, metrics: dict[str, float]) -> None:
        save_checkpoint(
            self.output_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            best_metric=self.best_metric,
            cfg=self.cfg,
            class_names=self.class_names,
            extra={"last_val_metrics": metrics},
        )

    def _resume(self, path: str) -> None:
        payload = load_checkpoint(path)
        self.model.load_state_dict(payload["model_state_dict"])
        if payload.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.epoch = int(payload.get("epoch", 0))
        self.best_metric = float(payload.get("best_metric", -float("inf")))
        logger.info(
            "resumed from %s at epoch %d (best %s=%.4f)", path, self.epoch, self.cfg.train.best_metric, self.best_metric
        )
