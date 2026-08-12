"""Tests for manifest building / loading (portable data pipeline)."""

import csv
import hashlib
import shutil

import pytest
import yaml
from PIL import Image

from garbage_classifier.data.manifest import ManifestError, build_manifest, load_manifest, manifest_root, validate_image


def _make_dataset(tmp_path, per_class: dict[str, int]) -> None:
    for cls_index, (cls, n) in enumerate(per_class.items()):
        d = tmp_path / "data" / cls
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (16, 16), color=(i * 10 % 255, cls_index * 40, 0)).save(d / f"{cls}{i}.jpg")


def _manifest_rows(manifests: dict[str, object]) -> dict[str, list[dict[str, str]]]:
    return {split: list(csv.DictReader(path.open())) for split, path in manifests.items()}


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
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "nope.csv", tmp_path)


def test_cross_class_duplicate_is_annotation_conflict(tmp_path):
    _make_dataset(tmp_path, {"paper": 1, "plastic": 1})
    paper = tmp_path / "data" / "paper" / "paper0.jpg"
    plastic = tmp_path / "data" / "plastic" / "plastic0.jpg"
    shutil.copyfile(paper, plastic)

    with pytest.raises(ManifestError) as exc_info:
        build_manifest(tmp_path / "data", tmp_path / "manifests")

    message = str(exc_info.value)
    assert "annotation conflict" in message
    assert "paper/paper0.jpg" in message
    assert "plastic/plastic0.jpg" in message
    assert "paper" in message
    assert "plastic" in message


def test_same_class_duplicates_are_assigned_to_one_split(tmp_path):
    _make_dataset(tmp_path, {"paper": 10})
    source = tmp_path / "data" / "paper" / "paper0.jpg"
    shutil.copyfile(source, source.with_name("paper-copy.jpg"))

    manifests = build_manifest(tmp_path / "data", tmp_path / "manifests", seed=19)
    rows = _manifest_rows(manifests)
    duplicate_splits = {
        split
        for split, split_rows in rows.items()
        if {row["path"] for row in split_rows} & {"paper/paper0.jpg", "paper/paper-copy.jpg"}
    }

    assert len(duplicate_splits) == 1
    split = duplicate_splits.pop()
    paths = {row["path"] for row in rows[split]}
    assert {"paper/paper0.jpg", "paper/paper-copy.jpg"} <= paths


def test_no_content_hash_crosses_splits(tmp_path):
    _make_dataset(tmp_path, {"paper": 12, "plastic": 12})
    for cls in ("paper", "plastic"):
        source = tmp_path / "data" / cls / f"{cls}0.jpg"
        shutil.copyfile(source, source.with_name(f"{cls}-copy.jpg"))

    manifests = build_manifest(tmp_path / "data", tmp_path / "manifests", seed=7)
    hashes_by_split = {}
    for split, rows in _manifest_rows(manifests).items():
        hashes_by_split[split] = {
            hashlib.sha256((tmp_path / "data" / row["path"]).read_bytes()).hexdigest() for row in rows
        }

    split_names = sorted(hashes_by_split)
    for index, split in enumerate(split_names):
        for other in split_names[index + 1 :]:
            assert hashes_by_split[split].isdisjoint(hashes_by_split[other])


def test_grouped_split_minimizes_target_count_deviation(tmp_path):
    class_dir = tmp_path / "data" / "paper"
    class_dir.mkdir(parents=True)
    for group_index, group_size in enumerate((2, 3, 5)):
        source = class_dir / f"{group_index}0.jpg"
        Image.new("RGB", (16, 16), color=(group_index * 80, 0, 0)).save(source)
        for copy_index in range(1, group_size):
            shutil.copyfile(source, class_dir / f"{group_index}{copy_index}.jpg")

    manifests = build_manifest(
        tmp_path / "data",
        tmp_path / "manifests",
        split_ratios=[0.8, 0.1, 0.1],
        seed=5,
    )
    counts = {split: len(rows) for split, rows in _manifest_rows(manifests).items()}

    assert counts == {"train": 8, "valid": 2}
    assert sum(abs(counts.get(split, 0) - target) for split, target in {"train": 8, "valid": 1, "test": 1}.items()) == 2


def test_strict_mode_rejects_same_class_duplicates(tmp_path):
    _make_dataset(tmp_path, {"paper": 2})
    source = tmp_path / "data" / "paper" / "paper0.jpg"
    duplicate = source.with_name("paper-copy.jpg")
    shutil.copyfile(source, duplicate)

    with pytest.raises(ManifestError, match="duplicate image content") as exc_info:
        build_manifest(tmp_path / "data", tmp_path / "manifests", strict=True)

    assert "paper/paper0.jpg" in str(exc_info.value)
    assert "paper/paper-copy.jpg" in str(exc_info.value)


def test_source_records_relative_data_root_schema(tmp_path):
    _make_dataset(tmp_path, {"paper": 3})
    manifest_dir = tmp_path / "metadata" / "manifests"

    build_manifest(tmp_path / "data", manifest_dir)

    source = yaml.safe_load((manifest_dir / "source.yaml").read_text())
    assert source["schema_version"] == 1
    assert source["data_root"] == {"path": "../../data", "relative_to": "manifest_dir"}
    assert "data_dir" not in source
    assert manifest_root(manifest_dir) == (tmp_path / "data").resolve()


def test_manifest_root_reads_legacy_absolute_data_dir(tmp_path):
    data_dir = tmp_path / "legacy-data"
    data_dir.mkdir()
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "source.yaml").write_text(yaml.safe_dump({"data_dir": str(data_dir.resolve())}))

    assert manifest_root(manifest_dir) == data_dir.resolve()


def test_dataset_root_override_supports_moved_data(tmp_path):
    pytest.importorskip("torch")
    from garbage_classifier.data.dataset import ImageClassificationDataset

    _make_dataset(tmp_path, {"paper": 3})
    manifest_dir = tmp_path / "manifests"
    manifests = build_manifest(tmp_path / "data", manifest_dir)
    moved_root = tmp_path / "moved-data"
    shutil.copytree(tmp_path / "data", moved_root)
    shutil.rmtree(tmp_path / "data")

    dataset = ImageClassificationDataset(manifests["test"], root_dir=moved_root)

    assert len(dataset) == 1
    assert dataset.samples[0][0].startswith(str(moved_root.resolve()))
