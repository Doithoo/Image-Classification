"""Tests for manifest building / loading (portable data pipeline)."""

from PIL import Image

from garbage_classifier.data.manifest import build_manifest, load_manifest, manifest_root, validate_image


def _make_dataset(tmp_path, per_class: dict[str, int]) -> None:
    for cls, n in per_class.items():
        d = tmp_path / "data" / cls
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (16, 16), color=(i * 10 % 255, 0, 0)).save(d / f"{cls}{i}.jpg")


def test_build_and_load_manifest(tmp_path):
    _make_dataset(tmp_path, {"a": 10, "b": 20})
    manifests = build_manifest(tmp_path / "data", tmp_path / "manifests", split_ratios=[0.8, 0.1, 0.1], seed=7)
    assert set(manifests) == {"train", "valid", "test"}

    train = load_manifest(manifests["train"], manifest_root(tmp_path / "manifests"))
    assert len(train) == 24  # 8 + 16
    # paths are absolute and exist
    import os

    assert all(os.path.exists(p) for p, _ in train)
    # labels match class order (alphabetical)
    assert {p.split("/")[-2] for p, _ in train} == {"a", "b"}


def test_split_is_stratified_and_deterministic(tmp_path):
    _make_dataset(tmp_path, {"a": 10, "b": 10, "c": 10})
    m1 = build_manifest(tmp_path / "data", tmp_path / "m1", seed=42)
    m2 = build_manifest(tmp_path / "data", tmp_path / "m2", seed=42)
    assert m1["train"].read_text() == m2["train"].read_text()
    # each class keeps 8/1/1
    from collections import Counter

    rows = load_manifest(m1["train"], manifest_root(tmp_path / "m1"))
    per_class = Counter(p.split("/")[-2] for p, _ in rows)
    assert per_class == {"a": 8, "b": 8, "c": 8}


def test_corrupt_image_detected(tmp_path):
    _make_dataset(tmp_path, {"a": 3})
    bad = tmp_path / "data" / "a" / "broken.jpg"
    bad.write_bytes(b"not an image at all")
    import pytest

    from garbage_classifier.data.manifest import ManifestError

    with pytest.raises(ManifestError):
        build_manifest(tmp_path / "data", tmp_path / "manifests", validate=True)
    assert validate_image(bad) is False


def test_missing_manifest_raises(tmp_path):
    import pytest

    from garbage_classifier.data.manifest import ManifestError

    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "nope.csv", tmp_path)
