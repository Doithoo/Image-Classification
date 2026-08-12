"""Class-imbalance handling: loss weights and sampling strategies.

Three comparable configurations are supported (roadmap item "imbalance
ablations"):

- ``none``      — plain CrossEntropyLoss / uniform sampling (baseline)
- ``inverse``   — loss weight w_i = N / (C * n_i), normalized (sums to 1)
- ``effective`` — class-balanced weights from Cui et al. 2019
                 (``Class-Balanced Loss Based on Effective Number of Samples``):
                 w_i = (1 - beta) / (1 - beta**n_i), beta = 0.999
"""

from __future__ import annotations


def compute_class_weights(counts: list[int], method: str = "none") -> list[float] | None:
    """Return per-class loss weights for ``method``, or None for 'none'."""
    if method == "none":
        return None
    n = len(counts)
    if n == 0 or any(c <= 0 for c in counts):
        raise ValueError(f"invalid class counts: {counts}")

    if method == "inverse":
        total = sum(counts)
        weights = [total / (n * c) for c in counts]
    elif method == "effective":
        beta = 0.999
        weights = [(1.0 - beta) / (1.0 - beta**c) for c in counts]
    else:
        raise ValueError(f"unknown class_weight method: {method!r} (none|inverse|effective)")

    # normalize so weights sum to the number of classes (stable loss scale)
    s = sum(weights)
    return [w * n / s for w in weights]


def build_weighted_sampler(labels: list[int], counts: list[int]):
    """WeightedRandomSampler with weight = 1/count per sample (oversamples rare classes)."""
    import torch

    weights = [1.0 / counts[label] for label in labels]
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
