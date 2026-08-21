"""Portable, auditable manifests for reproducible image classification."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from ..utils import file_sha256, publish_directory, write_text_atomic

DATASET_SCHEMA_VERSION = 2
_SPLITS = ("train", "valid", "test")


class ManifestError(RuntimeError):
    """Raised when prepared data is missing, corrupt or inconsistent."""


@dataclass(frozen=True)
class DatasetMetadata:
    """Versioned identity and layout contract for one prepared dataset."""

    schema_version: int
    data_root: dict[str, str | None]
    classes: tuple[str, ...]
    class_index: dict[str, int]
    split_counts: dict[str, int]
    per_class_counts: dict[str, dict[str, int]]
    manifest_sha256: dict[str, str]
    source_sha256: str
    identity: str
    seed: int
    split_ratios: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_root": self.data_root,
            "classes": list(self.classes),
            "class_index": self.class_index,
            "split_counts": self.split_counts,
            "per_class_counts": self.per_class_counts,
            "manifest_sha256": self.manifest_sha256,
            "source_sha256": self.source_sha256,
            "identity": self.identity,
            "seed": self.seed,
            "split_ratios": list(self.split_ratios),
        }


def _relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _iter_images(data_dir: Path, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[tuple[str, Path]]:
    samples: list[tuple[str, Path]] = []
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image in sorted(path for path in class_dir.iterdir() if path.suffix.lower() in extensions):
            if not image.name.startswith("._"):
                samples.append((class_dir.name, image))
    return samples


def validate_image(path: Path) -> bool:
    """Return whether Pillow can verify the image without decoding every pixel."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _content_groups(samples: list[tuple[str, Path]]) -> dict[str, list[tuple[str, Path]]]:
    groups: dict[str, list[tuple[str, Path]]] = {}
    for class_name, path in samples:
        groups.setdefault(file_sha256(path), []).append((class_name, path))
    return groups


def find_duplicates(data_dir: str | Path, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[list[str]]:
    """Return content-identical image paths grouped by SHA-256 digest."""
    root = Path(data_dir).resolve()
    groups = _content_groups(_iter_images(root, extensions))
    return [[str(path) for _class_name, path in group] for group in groups.values() if len(group) > 1]


def _describe_group(group: list[tuple[str, Path]], root: Path) -> str:
    return ", ".join(f"{_relpath(path, root)} (class {class_name!r})" for class_name, path in group)


def _split_groups(groups: list[list[Path]], split_ratios: list[float], rng: random.Random) -> dict[str, list[Path]]:
    """Assign indivisible duplicate groups with minimum target-count deviation."""
    shuffled = list(groups)
    rng.shuffle(shuffled)
    total = sum(len(group) for group in shuffled)
    targets = {
        "train": int(total * split_ratios[0]),
        "valid": int(total * (split_ratios[0] + split_ratios[1])) - int(total * split_ratios[0]),
    }
    targets["test"] = total - targets["train"] - targets["valid"]

    if all(len(group) == 1 for group in shuffled):
        train_end = targets["train"]
        valid_end = train_end + targets["valid"]
        return {
            "train": [path for group in shuffled[:train_end] for path in group],
            "valid": [path for group in shuffled[train_end:valid_end] for path in group],
            "test": [path for group in shuffled[valid_end:] for path in group],
        }

    duplicates = sorted((group for group in shuffled if len(group) > 1), key=len, reverse=True)
    singletons = [group for group in shuffled if len(group) == 1]
    states: dict[tuple[int, int], int] = {(0, 0): 0}
    for group in duplicates:
        size = len(group)
        next_states: dict[tuple[int, int], int] = {}
        for (train_count, valid_count), encoded_choices in states.items():
            for split_index, state in enumerate(
                ((train_count + size, valid_count), (train_count, valid_count + size), (train_count, valid_count))
            ):
                next_states.setdefault(state, encoded_choices * 3 + split_index)
        states = next_states

    duplicate_total = sum(len(group) for group in duplicates)

    def complete_counts(train_count: int, valid_count: int) -> tuple[int, int, int]:
        counts = [train_count, valid_count, duplicate_total - train_count - valid_count]
        remaining = len(singletons)
        for index, split in enumerate(_SPLITS):
            assigned = min(remaining, max(0, targets[split] - counts[index]))
            counts[index] += assigned
            remaining -= assigned
        counts[0] += remaining
        return counts[0], counts[1], counts[2]

    def score(state: tuple[int, int]) -> tuple[int, int, int, int, int]:
        counts = complete_counts(*state)
        deviations = tuple(abs(counts[index] - targets[split]) for index, split in enumerate(_SPLITS))
        return (
            sum(deviations),
            sum(max(0, counts[index] - targets[split]) for index, split in enumerate(_SPLITS)),
            deviations[0],
            deviations[1],
            deviations[2],
        )

    best_state = min(states, key=score)
    encoded = states[best_state]
    split_choices: list[int] = []
    for _group in reversed(duplicates):
        split_choices.append(encoded % 3)
        encoded //= 3
    split_choices.reverse()

    assigned: dict[str, list[Path]] = {split: [] for split in _SPLITS}
    for group, split_index in zip(duplicates, split_choices, strict=True):
        assigned[_SPLITS[split_index]].extend(group)
    targets_for_singletons = [
        complete_counts(*best_state)[index] - len(assigned[split]) for index, split in enumerate(_SPLITS)
    ]
    offset = 0
    for split, count in zip(_SPLITS, targets_for_singletons, strict=True):
        selected = singletons[offset : offset + count]
        assigned[split].extend(path for group in selected for path in group)
        offset += count
    return assigned


def _data_root_metadata(root: Path, final_manifest_dir: Path) -> dict[str, str | None]:
    try:
        relative = os.path.relpath(root, final_manifest_dir.resolve())
        return {"path": Path(relative).as_posix(), "relative_to": "manifest_dir"}
    except ValueError:
        return {"path": str(root), "relative_to": None}


def _source_sha256(root: Path, samples: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for class_name, path in sorted(samples, key=lambda item: _relpath(item[1], root)):
        digest.update(f"{class_name}\0{_relpath(path, root)}\0{file_sha256(path)}\n".encode())
    return digest.hexdigest()


def _identity_payload(
    *,
    classes: list[str],
    class_index: dict[str, int],
    split_counts: dict[str, int],
    per_class_counts: dict[str, dict[str, int]],
    manifest_sha256: dict[str, str],
    source_sha256: str,
    seed: int,
    split_ratios: list[float],
) -> str:
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "classes": classes,
        "class_index": class_index,
        "split_counts": split_counts,
        "per_class_counts": per_class_counts,
        "manifest_sha256": manifest_sha256,
        "source_sha256": source_sha256,
        "seed": seed,
        "split_ratios": split_ratios,
    }
    return hashlib.sha256(yaml.safe_dump(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _csv_text(rows: list[tuple[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["path", "label"])
    writer.writerows(rows)
    return output.getvalue()


def _write_summary(stage: Path, metadata: DatasetMetadata) -> None:
    lines = [
        f"schema_version={metadata.schema_version}",
        f"identity={metadata.identity}",
        f"source_sha256={metadata.source_sha256}",
        f"seed={metadata.seed}",
        f"split_ratios={list(metadata.split_ratios)}",
        f"classes={list(metadata.classes)}",
        f"class_index={metadata.class_index}",
    ]
    lines.extend(f"{split}_manifest_sha256={metadata.manifest_sha256[split]}" for split in _SPLITS)
    lines.extend(f"{split}={metadata.split_counts[split]}" for split in _SPLITS)
    write_text_atomic(stage / "summary.txt", "\n".join(lines) + "\n")


def build_manifest(
    data_dir: str | Path,
    manifest_dir: str | Path,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 666,
    validate: bool = True,
    strict: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create all split manifests and atomically publish their auditable identity."""
    root = Path(data_dir).resolve()
    destination = Path(manifest_dir)
    ratios = list(split_ratios)
    if len(ratios) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) for value in ratios
    ):
        raise ValueError("split_ratios must contain three numeric values")
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split_ratios must be non-negative and sum to 1")
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"manifest destination is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"manifest directory already exists and is non-empty: {destination}; use --overwrite")
    samples = _iter_images(root)
    if not samples:
        raise ManifestError(f"no images found under {root}")
    if validate:
        unreadable = [str(path) for _class_name, path in samples if not validate_image(path)]
        if unreadable:
            raise ManifestError(f"{len(unreadable)} unreadable images, e.g. {unreadable[:3]}")

    classes = sorted({class_name for class_name, _path in samples})
    class_index = {class_name: index for index, class_name in enumerate(classes)}
    groups = _content_groups(samples)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    for group in duplicate_groups:
        if len({class_name for class_name, _path in group}) > 1:
            raise ManifestError(
                "annotation conflict: identical image content has multiple classes: " + _describe_group(group, root)
            )
    if strict and duplicate_groups:
        raise ManifestError(
            "duplicate image content found in strict mode: " + _describe_group(duplicate_groups[0], root)
        )

    per_class: dict[str, list[list[Path]]] = {class_name: [] for class_name in classes}
    for group in groups.values():
        per_class[group[0][0]].append([path for _class_name, path in group])
    rng = random.Random(seed)
    rows_by_split: dict[str, list[tuple[str, str]]] = {split: [] for split in _SPLITS}
    for class_name, class_groups in per_class.items():
        for split, paths in _split_groups(class_groups, ratios, rng).items():
            rows_by_split[split].extend((_relpath(path, root), str(class_index[class_name])) for path in paths)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        for split in _SPLITS:
            write_text_atomic(stage / f"{split}.csv", _csv_text(rows_by_split[split]))
        manifest_sha256 = {split: file_sha256(stage / f"{split}.csv") for split in _SPLITS}
        split_counts = {split: len(rows_by_split[split]) for split in _SPLITS}
        per_class_counts = {split: {class_name: 0 for class_name in classes} for split in _SPLITS}
        for split, rows in rows_by_split.items():
            for _path, raw_label in rows:
                per_class_counts[split][classes[int(raw_label)]] += 1
        source_sha256 = _source_sha256(root, samples)
        identity = _identity_payload(
            classes=classes,
            class_index=class_index,
            split_counts=split_counts,
            per_class_counts=per_class_counts,
            manifest_sha256=manifest_sha256,
            source_sha256=source_sha256,
            seed=seed,
            split_ratios=ratios,
        )
        metadata = DatasetMetadata(
            schema_version=DATASET_SCHEMA_VERSION,
            data_root=_data_root_metadata(root, destination),
            classes=tuple(classes),
            class_index=class_index,
            split_counts=split_counts,
            per_class_counts=per_class_counts,
            manifest_sha256=manifest_sha256,
            source_sha256=source_sha256,
            identity=identity,
            seed=seed,
            split_ratios=(float(ratios[0]), float(ratios[1]), float(ratios[2])),
        )
        write_text_atomic(stage / "dataset.yaml", yaml.safe_dump(metadata.to_dict(), sort_keys=False))
        # Retain this minimal compatibility file for manifests produced before schema v2.
        write_text_atomic(
            stage / "source.yaml",
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "data_root": metadata.data_root,
                    "classes": list(metadata.classes),
                    "class_index": metadata.class_index,
                },
                sort_keys=False,
            ),
        )
        _write_summary(stage, metadata)
        publish_directory(stage, destination, overwrite=overwrite)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {split: destination / f"{split}.csv" for split in _SPLITS}


def _parse_metadata(raw: Any, path: Path) -> DatasetMetadata:
    if not isinstance(raw, dict) or raw.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ManifestError(f"unsupported dataset metadata in {path}; regenerate with prepare-data")
    try:
        classes = tuple(raw["classes"])
        class_index = dict(raw["class_index"])
        split_counts = {split: int(raw["split_counts"][split]) for split in _SPLITS}
        per_class_counts = {
            split: {str(name): int(count) for name, count in raw["per_class_counts"][split].items()}
            for split in _SPLITS
        }
        checksums = {split: str(raw["manifest_sha256"][split]) for split in _SPLITS}
        data_root = dict(raw["data_root"])
        ratios_raw = tuple(float(value) for value in raw["split_ratios"])
        if len(ratios_raw) != 3:
            raise ValueError("split_ratios must contain exactly three values")
        ratios: tuple[float, float, float] = (ratios_raw[0], ratios_raw[1], ratios_raw[2])
        metadata = DatasetMetadata(
            schema_version=int(raw["schema_version"]),
            data_root={"path": str(data_root["path"]), "relative_to": data_root.get("relative_to")},
            classes=classes,
            class_index={str(name): int(index) for name, index in class_index.items()},
            split_counts=split_counts,
            per_class_counts=per_class_counts,
            manifest_sha256=checksums,
            source_sha256=str(raw["source_sha256"]),
            identity=str(raw["identity"]),
            seed=int(raw["seed"]),
            split_ratios=ratios,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"invalid dataset metadata in {path}: {exc}") from exc
    if len(metadata.classes) == 0 or len(set(metadata.classes)) != len(metadata.classes):
        raise ManifestError(f"invalid class list in {path}")
    if metadata.class_index != {name: index for index, name in enumerate(metadata.classes)}:
        raise ManifestError(f"invalid class index in {path}")
    if len(metadata.split_ratios) != 3 or not metadata.identity:
        raise ManifestError(f"invalid split metadata in {path}")
    return metadata


def load_dataset_metadata(manifest_dir: str | Path) -> DatasetMetadata:
    """Load schema-v2 metadata; legacy manifests remain readable through helpers only."""
    path = Path(manifest_dir) / "dataset.yaml"
    if not path.is_file():
        raise ManifestError(f"missing {path}; regenerate manifests with prepare-data")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"invalid YAML in {path}: {exc}") from exc
    return _parse_metadata(raw, path)


def _root_from_data_root(manifest_dir: Path, data_root: dict[str, str | None]) -> Path:
    root_path = data_root.get("path")
    if not isinstance(root_path, str):
        raise ManifestError(f"invalid data_root path in {manifest_dir}")
    raw_path = Path(root_path)
    relative_to = data_root.get("relative_to")
    if relative_to == "manifest_dir":
        return (manifest_dir / raw_path).resolve()
    if relative_to is None and raw_path.is_absolute():
        return raw_path.resolve()
    raise ManifestError(f"unsupported data_root schema in {manifest_dir}: relative_to={relative_to!r}")


def manifest_classes(manifest_dir: str | Path) -> list[str]:
    """Return the ordered classes recorded by prepared metadata."""
    metadata_path = Path(manifest_dir) / "dataset.yaml"
    if metadata_path.is_file():
        return list(load_dataset_metadata(manifest_dir).classes)
    source = Path(manifest_dir) / "source.yaml"
    if not source.is_file():
        raise ManifestError(f"missing {metadata_path}; regenerate manifests with prepare-data")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ManifestError(f"invalid legacy metadata in {source}")
    classes = raw.get("classes")
    if not isinstance(classes, list) or not all(isinstance(name, str) and name for name in classes):
        raise ManifestError(f"no valid class list recorded in {source}")
    return list(classes)


def manifest_root(manifest_dir: str | Path) -> Path:
    """Return the source data root recorded by prepared metadata."""
    directory = Path(manifest_dir)
    if (directory / "dataset.yaml").is_file():
        return _root_from_data_root(directory, load_dataset_metadata(directory).data_root)
    source = directory / "source.yaml"
    if not source.is_file():
        raise ManifestError(f"missing {source}; regenerate manifests with prepare-data")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ManifestError(f"invalid legacy metadata in {source}")
    if "data_root" in raw and isinstance(raw["data_root"], dict):
        return _root_from_data_root(directory, raw["data_root"])
    if "data_dir" in raw:
        return Path(raw["data_dir"]).resolve()
    raise ManifestError(f"no data root recorded in {source}")


def load_manifest(
    manifest_path: str | Path, root_dir: str | Path, num_classes: int | None = None
) -> list[tuple[str, int]]:
    """Load validated relative paths and zero-based labels from one split CSV."""
    root = Path(root_dir).resolve()
    path = Path(manifest_path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    rows: list[tuple[str, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "label"]:
            raise ManifestError(f"manifest {path} must have exactly path,label columns")
        for row in reader:
            relpath = row.get("path")
            raw_label = row.get("label")
            if not relpath or raw_label is None:
                raise ManifestError(f"invalid manifest row in {path}: {row}")
            absolute = (root / relpath).resolve()
            try:
                absolute.relative_to(root)
            except ValueError as exc:
                raise ManifestError(f"manifest entry outside data root: {relpath!r}; root={root}") from exc
            if not absolute.is_file():
                raise ManifestError(f"manifest entry missing on disk: {absolute}")
            try:
                label = int(raw_label)
            except ValueError as exc:
                raise ManifestError(f"manifest label is not an integer: {raw_label!r}") from exc
            if label < 0 or (num_classes is not None and label >= num_classes):
                raise ManifestError(f"manifest label out of range: {label}")
            rows.append((str(absolute), label))
    return rows


def verify_prepared_data(
    manifest_dir: str | Path,
    data_dir: str | Path | None = None,
) -> DatasetMetadata:
    """Verify prepared CSVs, labels, source bytes and the combined dataset identity."""
    directory = Path(manifest_dir)
    metadata = load_dataset_metadata(directory)
    root = Path(data_dir).resolve() if data_dir is not None else manifest_root(directory)
    samples: list[tuple[str, Path]] = []
    for split in _SPLITS:
        path = directory / f"{split}.csv"
        if not path.is_file():
            raise ManifestError(f"missing manifest split: {path}")
        if file_sha256(path) != metadata.manifest_sha256[split]:
            raise ManifestError(f"manifest checksum mismatch: {path}")
        rows = load_manifest(path, root, len(metadata.classes))
        if len(rows) != metadata.split_counts[split]:
            raise ManifestError(f"manifest count mismatch: {path}")
        counts = Counter(label for _image, label in rows)
        expected_counts = {name: metadata.per_class_counts[split][name] for name in metadata.classes}
        actual_counts = {name: counts[index] for index, name in enumerate(metadata.classes)}
        if actual_counts != expected_counts:
            raise ManifestError(f"per-class count mismatch: {path}")
    for class_name, path in _iter_images(root):
        samples.append((class_name, path))
    if _source_sha256(root, samples) != metadata.source_sha256:
        raise ManifestError("source image checksum mismatch; regenerate manifests after reviewing data changes")
    expected_identity = _identity_payload(
        classes=list(metadata.classes),
        class_index=metadata.class_index,
        split_counts=metadata.split_counts,
        per_class_counts=metadata.per_class_counts,
        manifest_sha256=metadata.manifest_sha256,
        source_sha256=metadata.source_sha256,
        seed=metadata.seed,
        split_ratios=list(metadata.split_ratios),
    )
    if metadata.identity != expected_identity:
        raise ManifestError("dataset identity does not match its recorded metadata")
    return metadata


def inspect_prepared_data(manifest_dir: str | Path, data_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a concise, machine-readable prepared-data summary after validation."""
    metadata = verify_prepared_data(manifest_dir, data_dir)
    return {
        "schema_version": metadata.schema_version,
        "identity": metadata.identity,
        "classes": list(metadata.classes),
        "split_counts": metadata.split_counts,
        "per_class_counts": metadata.per_class_counts,
        "source_sha256": metadata.source_sha256,
    }
