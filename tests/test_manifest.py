"""Prepared-data identity, duplicate protection and path-validation tests."""

import csv
import hashlib
import shutil

import pytest
import yaml
from PIL import Image

from garbage_classifier.data.manifest import (
    DATASET_SCHEMA_VERSION,
    ManifestError,
    build_manifest,
    inspect_prepared_data,
    load_dataset_metadata,
    load_manifest,
    manifest_root,
    validate_image,
    verify_prepared_data,
)


def _make_dataset(tmp_path, per_class: dict[str, int]) -> None:
    for class_index, (class_name, count) in enumerate(per_class.items()):
        directory = tmp_path / "data" / class_name
        directory.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (16, 16), color=(index * 10 % 255, class_index * 40, 0)).save(
                directory / f"{class_name}{index}.jpg"
            )


def _manifest_rows(manifests: dict[str, object]) -> dict[str, list[dict[str, str]]]:
    return {split: list(csv.DictReader(path.open())) for split, path in manifests.items()}


def test_build_verify_and_inspect_versioned_manifest(tmp_path):
    _make_dataset(tmp_path, {"a": 10, "b": 20})
    manifests = build_manifest(tmp_path / "data", tmp_path / "manifests", split_ratios=[0.8, 0.1, 0.1], seed=7)

    metadata = verify_prepared_data(tmp_path / "manifests")
    report = inspect_prepared_data(tmp_path / "manifests")

    assert set(manifests) == {"train", "valid", "test"}
    assert metadata.schema_version == DATASET_SCHEMA_VERSION
    assert metadata.classes == ("a", "b")
    assert len(metadata.identity) == 64
    assert report["identity"] == metadata.identity
    assert metadata.split_counts == {"train": 24, "valid": 3, "test": 3}
    assert metadata.manifest_sha256["train"] == hashlib.sha256(manifests["train"].read_bytes()).hexdigest()
    train = load_manifest(manifests["train"], manifest_root(tmp_path / "manifests"), num_classes=2)
    assert all(path.startswith(str((tmp_path / "data").resolve())) for path, _label in train)


def test_identity_is_deterministic_and_bind_all_splits(tmp_path):
    _make_dataset(tmp_path, {"a": 10, "b": 10, "c": 10})
    build_manifest(tmp_path / "data", tmp_path / "m1", seed=42)
    build_manifest(tmp_path / "data", tmp_path / "m2", seed=42)
    first = load_dataset_metadata(tmp_path / "m1")
    second = load_dataset_metadata(tmp_path / "m2")

    assert first.identity == second.identity
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.per_class_counts["train"] == {"a": 8, "b": 8, "c": 8}


def test_verify_detects_modified_manifest_and_source_bytes(tmp_path):
    _make_dataset(tmp_path, {"a": 10, "b": 10})
    build_manifest(tmp_path / "data", tmp_path / "manifests")
    manifest = tmp_path / "manifests" / "train.csv"
    manifest.write_text(manifest.read_text() + "a/a0.jpg,0\n")
    with pytest.raises(ManifestError, match="checksum mismatch"):
        verify_prepared_data(tmp_path / "manifests")

    build_manifest(tmp_path / "data", tmp_path / "manifests", overwrite=True)
    Image.new("RGB", (16, 16), color="white").save(tmp_path / "data" / "a" / "a0.jpg")
    with pytest.raises(ManifestError, match="source image checksum mismatch"):
        verify_prepared_data(tmp_path / "manifests")


def test_prepare_refuses_nonempty_destination_without_overwrite(tmp_path):
    _make_dataset(tmp_path, {"a": 4, "b": 4})
    destination = tmp_path / "manifests"
    build_manifest(tmp_path / "data", destination)
    with pytest.raises(FileExistsError, match="--overwrite"):
        build_manifest(tmp_path / "data", destination)
    build_manifest(tmp_path / "data", destination, overwrite=True)


def test_corrupt_and_cross_class_duplicate_images_are_rejected(tmp_path):
    _make_dataset(tmp_path, {"paper": 2, "plastic": 2})
    broken = tmp_path / "data" / "paper" / "broken.jpg"
    broken.write_bytes(b"not an image")
    with pytest.raises(ManifestError, match="unreadable"):
        build_manifest(tmp_path / "data", tmp_path / "manifests")
    assert not validate_image(broken)

    broken.unlink()
    shutil.copyfile(tmp_path / "data" / "paper" / "paper0.jpg", tmp_path / "data" / "plastic" / "plastic0.jpg")
    with pytest.raises(ManifestError, match="annotation conflict"):
        build_manifest(tmp_path / "data", tmp_path / "manifests")


def test_duplicates_remain_in_one_split_and_strict_mode_rejects_them(tmp_path):
    _make_dataset(tmp_path, {"paper": 10})
    source = tmp_path / "data" / "paper" / "paper0.jpg"
    duplicate = source.with_name("paper-copy.jpg")
    shutil.copyfile(source, duplicate)
    manifests = build_manifest(tmp_path / "data", tmp_path / "manifests", seed=19)
    rows = _manifest_rows(manifests)
    split_names = [
        split
        for split, split_rows in rows.items()
        if {row["path"] for row in split_rows} & {"paper/paper0.jpg", "paper/paper-copy.jpg"}
    ]
    assert len(split_names) == 1
    with pytest.raises(ManifestError, match="duplicate image content"):
        build_manifest(tmp_path / "data", tmp_path / "strict", strict=True)


def test_empty_split_is_recorded_and_can_be_read(tmp_path):
    directory = tmp_path / "data" / "paper"
    directory.mkdir(parents=True)
    for group_index, group_size in enumerate((2, 3, 5)):
        source = directory / f"{group_index}0.jpg"
        Image.new("RGB", (16, 16), color=(group_index * 80, 0, 0)).save(source)
        for copy_index in range(1, group_size):
            shutil.copyfile(source, directory / f"{group_index}{copy_index}.jpg")

    manifests = build_manifest(tmp_path / "data", tmp_path / "manifests", seed=5)
    counts = {split: len(rows) for split, rows in _manifest_rows(manifests).items()}
    assert counts == {"train": 8, "valid": 2, "test": 0}
    assert load_dataset_metadata(tmp_path / "manifests").split_counts == counts


def test_data_root_is_portable_and_legacy_source_schema_is_readable(tmp_path):
    _make_dataset(tmp_path, {"paper": 3})
    directory = tmp_path / "metadata" / "manifests"
    build_manifest(tmp_path / "data", directory)
    source = yaml.safe_load((directory / "source.yaml").read_text())
    assert source["schema_version"] == 1
    assert source["data_root"] == {"path": "../../data", "relative_to": "manifest_dir"}
    assert manifest_root(directory) == (tmp_path / "data").resolve()

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "source.yaml").write_text(yaml.safe_dump({"data_dir": str((tmp_path / "data").resolve())}))
    assert manifest_root(legacy) == (tmp_path / "data").resolve()


@pytest.mark.parametrize("entry", ["../outside.jpg", "absolute"])
def test_load_manifest_rejects_paths_outside_root_and_bad_labels(tmp_path, entry):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (16, 16)).save(outside)
    path = str(outside.resolve()) if entry == "absolute" else entry
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"path,label\n{path},3\n")
    with pytest.raises(ManifestError, match="outside data root"):
        load_manifest(manifest, root, num_classes=2)

    valid = root / "image.jpg"
    Image.new("RGB", (16, 16)).save(valid)
    manifest.write_text("path,label\nimage.jpg,2\n")
    with pytest.raises(ManifestError, match="label out of range"):
        load_manifest(manifest, root, num_classes=2)
