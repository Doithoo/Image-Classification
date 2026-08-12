"""Tests for class-imbalance handling (loss weights + weighted sampler)."""

import numpy as np

from garbage_classifier.training.weights import build_weighted_sampler, compute_class_weights


def test_none_returns_none():
    assert compute_class_weights([100, 10, 1], "none") is None


def test_inverse_weights():
    counts = [100, 100, 10]  # rare class gets the largest weight
    w = compute_class_weights(counts, "inverse")
    assert w is not None
    assert w[2] > w[0]
    assert np.isclose(sum(w), 3.0)  # normalized to number of classes


def test_effective_weights():
    counts = [100, 10]
    w = compute_class_weights(counts, "effective")
    assert w is not None
    assert w[1] > w[0]
    assert all(x > 0 for x in w)


def test_unknown_method_raises():
    import pytest

    with pytest.raises(ValueError):
        compute_class_weights([1, 2, 3], "bogus")


def test_weighted_sampler_oversamples_rare_class():
    counts = [90, 10]
    labels = [0] * 90 + [1] * 10
    sampler = build_weighted_sampler(labels, counts)
    indices = list(sampler)
    # with replacement and 100 samples, the rare class (10% of data) appears ~50% of the time
    rare_share = sum(1 for i in indices if labels[i] == 1) / len(indices)
    assert 0.3 < rare_share < 0.7
