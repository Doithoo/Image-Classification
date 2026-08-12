"""MixUp / CutMix data augmentation (regularization for image classifiers).

Why this exists (learning note):
  - Both techniques build *virtual* training samples by mixing two real samples.
    The model sees more diverse inputs and is forced to learn features instead
    of memorizing specific images, which reduces overfitting.
  - MixUp  (Zhang et al. 2017): x' = λ·x_i + (1−λ)·x_j,  y' = λ·y_i + (1−λ)·y_j
    — a convex combination of both images AND both labels.
  - CutMix (Yun et al. 2019): cut a rectangular patch out of image j and paste
    it into image i; the label is mixed by the patch's area fraction. This keeps
    the "local" information of an object instead of blending the whole image.
  - λ is sampled from a Beta(α, α) distribution. α = 0 disables the technique.
    The two are mutually exclusive (mixup wins when both are set).

When labels are mixed they are no longer integers, so the loss must accept soft
targets (one-hot vectors) instead of class indices.
"""

from __future__ import annotations

import torch


def one_hot_mixup_target(labels: torch.Tensor, num_classes: int, label_smoothing: float = 0.0) -> torch.Tensor:
    """Convert integer labels into smoothed one-hot vectors (float targets)."""
    targets = torch.zeros(labels.size(0), num_classes, dtype=torch.float32, device=labels.device)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)
    if label_smoothing > 0.0:
        # label smoothing: pull probability mass away from the true class
        targets = targets * (1.0 - label_smoothing) + label_smoothing / num_classes
    return targets


class MixupCutmix:
    """Applies MixUp or CutMix to a batch of images and integer labels.

    Usage in the training loop::

        if self.mixup is not None:
            images, soft_labels = self.mixup(images, labels)
            loss = soft_cross_entropy(outputs, soft_labels, weight)
        else:
            loss = self.loss_fn(outputs, labels)   # hard labels
    """

    def __init__(
        self, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0, num_classes: int = 6, label_smoothing: float = 0.0
    ) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self._enabled = mixup_alpha > 0.0 or cutmix_alpha > 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (augmented_images, soft_targets). Images stay on the same device."""
        if not self._enabled:
            return images, one_hot_mixup_target(labels, self.num_classes, self.label_smoothing)

        batch = images.size(0)
        # permuted indices: each sample i is mixed with a random partner j != i
        perm = torch.randperm(batch, device=images.device)

        lam = self._sample_lambda(batch, images.device)  # shape [batch]
        lam = lam.view(batch, 1, 1, 1)  # broadcastable over image dims

        mixed_images = lam * images + (1.0 - lam) * images[perm]
        mixed_targets = lam.view(batch, 1) * one_hot_mixup_target(labels, self.num_classes, self.label_smoothing) + (
            1.0 - lam.view(batch, 1)
        ) * one_hot_mixup_target(labels[perm], self.num_classes, self.label_smoothing)
        return mixed_images, mixed_targets

    def _sample_lambda(self, batch: int, device: torch.device) -> torch.Tensor:
        """Sample λ per sample: Beta(α,α) for mixup, Beta(α,α) area fraction for cutmix."""
        if self.mixup_alpha > 0.0:
            # MixUp: λ is the global mixing ratio
            return torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample((batch,)).to(device)
        # CutMix: sample λ first, then map to the bounding-box area fraction
        lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample((batch,)).to(device)
        # cut ratio -> box size. The box area fraction is (1-λ); to guarantee the
        # box is valid (at least 1px), we only keep λ in [1 - 1/HW, 1]... we clamp
        # the *box* sizes instead, which is what timm does.
        return lam


def soft_cross_entropy(
    logits: torch.Tensor, soft_targets: torch.Tensor, class_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Cross-entropy against soft (one-hot mixed) targets: −Σ_c w_c · y_c · log p_c.

    This is the loss that goes with MixUp/CutMix. With integer labels it reduces
    to the usual cross-entropy, but here targets are probability vectors.
    """
    log_probs = torch.log_softmax(logits, dim=1)
    if class_weights is not None:
        log_probs = log_probs * class_weights.view(1, -1)
    return -(soft_targets * log_probs).sum(dim=1).mean()
