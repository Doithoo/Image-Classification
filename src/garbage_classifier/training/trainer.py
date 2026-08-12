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
        class_weights: list[float] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        t = cfg.train
        if class_weights:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
            self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=t.label_smoothing)
            self.class_weights_tensor = weight_tensor
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)
            self.class_weights_tensor = None
        params = [p for p in model.parameters() if p.requires_grad]
        if t.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=t.lr, momentum=t.momentum, weight_decay=t.weight_decay)
        elif t.optimizer == "lion":
            from lion_pytorch import Lion

            self.optimizer = Lion(params, lr=t.lr, weight_decay=t.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)

        # LR schedule with optional linear warmup (see mixup.py for the rationale of warmup)
        # Warmup avoids large gradient updates at the very start, when the network
        # (or the fine-tuned head) is far from a good region of the loss landscape.
        self.scheduler: Any | None = None
        if t.scheduler in ("cosine", "step"):
            if t.scheduler == "cosine":
                base = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=max(1, t.epochs - t.warmup_epochs)
                )
            else:
                base = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, t.epochs // 3), gamma=0.1)
            if t.warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=t.warmup_epochs
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers=[warmup, base], milestones=[t.warmup_epochs]
                )
            else:
                self.scheduler = base

        # MixUp / CutMix augmentation (soft targets) and EMA shadow weights
        from .ema import EMA
        from .mixup import MixupCutmix

        self.mixup = MixupCutmix(
            mixup_alpha=t.mixup_alpha,
            cutmix_alpha=t.cutmix_alpha,
            num_classes=len(class_names),
            label_smoothing=t.label_smoothing,
        )
        self.ema: EMA | None = EMA(model, decay=t.ema_decay) if t.ema else None

        self.scaler = torch.amp.GradScaler("cuda", enabled=(t.amp and device.type == "cuda"))
        self.use_amp = t.amp and device.type in ("cuda", "mps")

        self.epoch = 0
        self.best_metric = -float("inf")
        self.patience_left = t.early_stop_patience

    # ---- public API -------------------------------------------------------
    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, resume_from: str | None = None) -> dict[str, Any]:
        """Run the full train/validate loop.

        Loop walkthrough (learning note):
          1. train one epoch -> per-step: forward, loss, backward, optimizer.step,
             (optional) EMA shadow update, (optional) MixUp soft targets
          2. (optional) swap in EMA shadow weights for validation
          3. validate -> collect predictions, compute metrics (accuracy,
             balanced accuracy, macro-F1, ...), log to metrics.csv
          4. checkpoint selection happens while EMA weights are applied, so
             best.pt stores the deployable (EMA) model
          5. restore fast weights (EMA only), step the LR scheduler, check early
             stopping
        """
        csv_path = self.output_dir / "metrics.csv"
        csv_logger = CsvLogger(csv_path, ["epoch", "train_loss", "val_loss", "accuracy", "balanced_acc", "macro_f1"])

        if resume_from is not None:
            self._resume(resume_from)

        start = time.time()
        for epoch in range(self.epoch, self.cfg.train.epochs):
            train_loss = self._run_epoch(train_loader, train=True)["loss"]
            if self.ema is not None:
                # EMA is updated per step inside _run_epoch; here we only swap in
                # the shadow weights for validation and restore the fast weights
                self._fast_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                self.ema.apply_to(self.model)
            metrics = self._run_epoch(valid_loader, train=False)

            deployable_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            if self.ema is not None:
                self.model.load_state_dict(self._fast_state)

            # checkpoint selection happens while EMA weights are applied (if enabled),
            # so best.pt/last.pt store the deployable (EMA) model
            self.epoch = epoch + 1
            score = metrics.get(self.cfg.train.best_metric, metrics["accuracy"])
            improved = score > self.best_metric + 1e-6
            if improved:
                self.best_metric = score
                self.patience_left = self.cfg.train.early_stop_patience
            else:
                self.patience_left -= 1
            if self.scheduler is not None:
                self.scheduler.step()
            if improved:
                self._save("best.pt", metrics, deployable_state)
            self._save("last.pt", metrics, deployable_state)

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
                if train and self.mixup.enabled:
                    # MixUp/CutMix produces soft (one-hot mixed) targets
                    images, soft_labels = self.mixup(images, labels)
                if train:
                    # 清空上一步的梯度；set_to_none 比 zero_ 更快且省内存
                    self.optimizer.zero_grad(set_to_none=True)

                # ---- 前向 + 损失 ----
                # autocast: 前向计算用 float16（省显存、更快），权重仍存 float32。
                # 这就是混合精度（AMP）：关键数值用高精度，中间计算用低精度。
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    outputs = self.model(images)  # 前向：模型对这批图的预测 logits
                    if train and self.mixup.enabled:
                        # MixUp 后标签是软标签（概率向量），普通 CrossEntropyLoss
                        # 只接受整数标签，所以要换软标签版本的损失函数
                        from .mixup import soft_cross_entropy

                        loss = soft_cross_entropy(outputs, soft_labels, self.class_weights_tensor)
                    else:
                        loss = self.loss_fn(outputs, labels)  # 比较预测与真相

                if train:
                    # ---- 反向 + 参数更新 ----
                    if self.use_amp and self.device.type == "cuda":
                        # CUDA 上 float16 梯度可能下溢，用 GradScaler 动态放大梯度；
                        # 更新前必须 unscale 还原，梯度裁剪也要在 unscale 之后。
                        self.scaler.scale(loss).backward()
                        if self.cfg.train.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                        self.scaler.step(self.optimizer)  # 若梯度为 NaN/Inf 则跳过本次更新
                        self.scaler.update()  # 调整下一次的放大倍数
                    else:
                        loss.backward()  # 反向传播：算每个参数的梯度
                        if self.cfg.train.grad_clip > 0:
                            # 梯度裁剪：把整组梯度范数限制在阈值内，防止梯度爆炸
                            # （常见于深层网络/Transformer）
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                        self.optimizer.step()  # 沿梯度更新参数：w -= lr * grad
                    if train and self.ema is not None:
                        # EMA shadows the weights after EVERY step (decay=0.999 ->
                        # ~6% of the trajectory per epoch at 63 steps/epoch)
                        self.ema.update(self.model)

                if not train:
                    all_preds.append(outputs.argmax(dim=1).detach().cpu())
                    all_labels.append(labels.detach().cpu())
                total_loss += loss.item()
                n_batches += 1

        if n_batches == 0:
            phase = "training" if train else "validation"
            raise ValueError(f"{phase} loader is empty; provide at least one sample")
        if train:
            return {"loss": total_loss / n_batches}
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        metrics = evaluate_predictions(preds, labels, num_classes=len(self.class_names))
        metrics["loss"] = total_loss / n_batches
        return metrics

    def _save(
        self, name: str, metrics: dict[str, float], deployable_state: dict[str, torch.Tensor] | None = None
    ) -> None:
        save_checkpoint(
            self.output_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            deployable_state_dict=deployable_state,
            ema=self.ema,
            scaler=self.scaler,
            patience_left=self.patience_left,
            epoch=self.epoch,
            best_metric=self.best_metric,
            cfg=self.cfg,
            class_names=self.class_names,
            extra={"last_val_metrics": metrics},
        )

    def _resume(self, path: str) -> None:
        payload = load_checkpoint(path)
        self.model.load_state_dict(payload["training_model_state_dict"])
        if payload.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.epoch = int(payload.get("epoch", 0))
        self.best_metric = float(payload.get("best_metric", -float("inf")))
        if self.ema is not None and payload.get("ema_state_dict") is not None:
            self.ema.load_state_dict(payload["ema_state_dict"])
        if payload.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(payload["scaler_state_dict"])
        if payload.get("patience_left") is not None:
            self.patience_left = int(payload["patience_left"])
        rng_state = payload.get("rng_state", {})
        if "python" in rng_state:
            import random

            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            import numpy as np

            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda"):
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        logger.info(
            "resumed from %s at epoch %d (best %s=%.4f)", path, self.epoch, self.cfg.train.best_metric, self.best_metric
        )
