"""Packaging, documentation topology and release-policy regression checks."""

import re
from dataclasses import fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).parents[1]


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_package_metadata_declares_release_and_project_links():
    project = _pyproject()["project"]
    assert project["requires-python"] == ">=3.10,<3.13"
    assert project["version"] == "1.9.0"
    assert "Development Status :: 4 - Beta" in project["classifiers"]
    assert "Intended Audience :: Education" in project["classifiers"]
    assert {"Homepage", "Repository", "Documentation", "Issues", "Security"} <= set(project["urls"])
    assert all(">=" in requirement and "<" in requirement for requirement in project["dependencies"])


def test_dev_tools_ci_and_distribution_manifest_cover_maintainer_contract():
    extras = _pyproject()["project"]["optional-dependencies"]
    for dependency in ("pytest", "ruff", "mypy", "pre-commit", "build", "twine"):
        assert any(requirement.startswith(dependency) for requirement in extras["dev"])
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    for command in ("uv sync --locked", "ruff check", "mypy", "pytest -W error::DeprecationWarning", "uv build"):
        assert command in workflow
    assert "permissions:" in workflow and "contents: read" in workflow
    assert "actions/checkout@3d3c42e5" in workflow
    manifest = (ROOT / "MANIFEST.in").read_text()
    for directory in ("configs", "docs", "examples", "scripts", "tests"):
        assert directory in manifest
    assert (ROOT / "SECURITY.md").is_file()


def test_markdown_relative_links_resolve_after_documentation_reorganization():
    link_pattern = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
    broken = []
    documents = [ROOT / "README.md", ROOT / "README.zh-CN.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for document in documents:
        for raw_target in link_pattern.findall(document.read_text()):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "/")):
                continue
            if not (document.parent / target).exists():
                broken.append(f"{document.relative_to(ROOT)} -> {raw_target}")
    assert not broken, "broken relative Markdown links:\n" + "\n".join(broken)


def test_docs_are_grouped_by_reader_intent_and_have_bilingual_entry_points():
    expected = {
        "docs/README.md",
        "docs/README.zh-CN.md",
        "docs/tutorial/README.md",
        "docs/concepts/classification-flow.md",
        "docs/guides/using-your-data.md",
        "docs/guides/troubleshooting.md",
        "docs/reference/config-reference.md",
        "docs/reference/dataset-format.md",
        "docs/reference/checkpoint-schema.md",
        "docs/architecture/0001-reproducible-classification-contracts.md",
        "docs/recorded-run/README.md",
        "configs/README.md",
        "examples/README.md",
    }
    assert all((ROOT / path).is_file() for path in expected)
    for english in (ROOT / "docs").rglob("*.md"):
        if english.name.endswith(".zh-CN.md") or english.name == "README.md" and english.parent == ROOT / "docs":
            continue
        chinese = english.with_name(f"{english.stem}.zh-CN.md")
        if english.parent.name in {"assets", "recorded-run"}:
            continue
        assert chinese.is_file(), f"missing Chinese counterpart for {english.relative_to(ROOT)}"


def test_root_readmes_keep_one_short_workflow_and_documentation_handoff():
    english = (ROOT / "README.md").read_text()
    chinese = (ROOT / "README.zh-CN.md").read_text()
    for readme in (english, chinese):
        for command in ("prepare-data", "verify-data", "garbage train", "garbage evaluate", "garbage predict"):
            assert command in readme
        assert "docs/README" in readme
    assert "## Start Here" in english
    assert "## 从这里开始" in chinese


def test_recorded_run_contains_machine_readable_historical_evidence():
    root = ROOT / "docs/recorded-run"
    for relative in (
        "README.md",
        "README.zh-CN.md",
        "config.yaml",
        "run.yaml",
        "metrics.csv",
        "evaluation/evaluation.json",
    ):
        assert (root / relative).is_file()
    assert "Historical schema-v1" in (root / "README.md").read_text()


def test_configuration_reference_covers_every_user_facing_field():
    from garbage_classifier.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig

    references = [
        (ROOT / "docs/reference/config-reference.md").read_text(),
        (ROOT / "docs/reference/config-reference.zh-CN.md").read_text(),
    ]
    keys = {
        *(f"data.{field.name}" for field in fields(DataConfig)),
        *(f"model.{field.name}" for field in fields(ModelConfig)),
        *(f"train.{field.name}" for field in fields(TrainConfig)),
        *(field.name for field in fields(ExperimentConfig) if field.name not in {"data", "model", "train"}),
    }
    for reference in references:
        assert not sorted(key for key in keys if f"`{key}`" not in reference)
