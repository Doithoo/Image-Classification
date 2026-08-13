"""Offline tests for the audited v1 dataset patch."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "download_data.py"
SPEC = importlib.util.spec_from_file_location("download_data", SCRIPT)
assert SPEC and SPEC.loader
download_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_data)


def _write_audited_files(root: Path) -> tuple[dict[str, str], dict[str, bytes]]:
    hashes: dict[str, str] = {}
    contents: dict[str, bytes] = {}
    for relative_path in download_data.DATASET_AUDIT_REMOVALS:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"fixture for {relative_path}".encode()
        contents[relative_path] = payload
        path.write_bytes(payload)
        hashes[relative_path] = hashlib.sha256(payload).hexdigest()
    return hashes, contents


def test_apply_dataset_audit_removes_only_hash_matched_files(tmp_path, monkeypatch):
    hashes, _ = _write_audited_files(tmp_path)
    monkeypatch.setattr(download_data, "DATASET_AUDIT_REMOVALS", hashes)
    keep = tmp_path / "paper" / "keep.jpg"
    keep.parent.mkdir(parents=True)
    keep.write_bytes(b"keep")

    removed = download_data.apply_dataset_audit(tmp_path)

    assert set(removed) == set(hashes)
    assert all(not (tmp_path / path).exists() for path in hashes)
    assert keep.read_bytes() == b"keep"


def test_apply_dataset_audit_fails_before_deleting_on_hash_mismatch(tmp_path, monkeypatch):
    expected, contents = _write_audited_files(tmp_path)
    expected["plastic/plastic152.jpg"] = "0" * 64
    monkeypatch.setattr(download_data, "DATASET_AUDIT_REMOVALS", expected)

    with pytest.raises(RuntimeError, match="plastic/plastic152.jpg.*checksum mismatch"):
        download_data.apply_dataset_audit(tmp_path)

    assert all((tmp_path / path).read_bytes() == payload for path, payload in contents.items())
    assert set(download_data.DATASET_AUDIT_REMOVALS) == set(contents)


def test_apply_dataset_audit_is_idempotent_when_targets_are_absent(tmp_path):
    assert download_data.apply_dataset_audit(tmp_path) == []
