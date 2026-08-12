"""Exponential Moving Average (EMA) of model weights.

Why this exists (learning note):
  - During training the weights jitter around; averaging them over time smooths
    the trajectory and typically lands in a flatter, more generalizable minimum.
  - EMA maintains a shadow copy: shadow ← decay·shadow + (1−decay)·weights,
    updated after every training step/epoch. At evaluation time we swap the EMA
    weights in, get the metrics, and (optionally) save them as the best model.
  - A decay of 0.999 means each update moves the shadow by only 0.1% — the shadow
    lags behind the fast weights but is much more stable.

Note: batch-norm running statistics (running_mean/var, num_batches_tracked) are
NOT averaged — they should stay with the fast model and be copied as-is when
applying the EMA weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_BN_KEYS = ("num_batches_tracked",)  # counter, not a learned statistic


class EMA:
    """Shadow-weight tracker. ``apply_to`` swaps EMA weights into the model.

    Learning note: batch-norm *running statistics* (running_mean/var) ARE part
    of the EMA here. If the shadow excluded them, validation would pair shadow
    weights with the fast model's BN statistics — during fast-changing phases
    (pretrained fine-tuning) the mismatch saturates activations and collapses
    predictions to a single class. Averaging running stats too keeps the shadow
    self-consistent (this is what timm's ``ModelEmaV2`` does).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.state_dict().items():
            if "num_batches_tracked" not in name:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Fold the current fast weights into the EMA shadow."""
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Load the EMA shadow into the model (for evaluation / saving)."""
        state = model.state_dict()
        for name, param in self.shadow.items():
            if name in state:
                state[name].copy_(param)
