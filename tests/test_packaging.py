import re
from dataclasses import fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_package_metadata_and_dependencies_are_release_ready():
    project = _pyproject()["project"]

    assert project["requires-python"] == ">=3.10,<3.13"
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert {
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    } <= set(project["classifiers"])
    assert all(">=" in requirement and "<" in requirement for requirement in project["dependencies"])
    assert not any(
        name in requirement
        for requirement in project["dependencies"]
        for name in ("tqdm", "matplotlib", "lion-pytorch")
    )


def test_optional_dependencies_cover_release_and_development_tools():
    extras = _pyproject()["project"]["optional-dependencies"]

    assert any(requirement.startswith("matplotlib") for requirement in extras["plot"])
    assert any(requirement.startswith("lion-pytorch") for requirement in extras["lion"])
    for command in ("pytest", "ruff", "pre-commit", "build", "twine", "matplotlib", "lion-pytorch"):
        assert any(requirement.startswith(command) for requirement in extras["dev"])
    assert set(extras["all"]) >= set(extras["onnx"] + extras["demo"] + extras["plot"] + extras["lion"])


def test_onnx_runtime_supports_every_declared_python_version():
    extras = _pyproject()["project"]["optional-dependencies"]

    for extra in ("onnx", "all"):
        runtimes = [requirement for requirement in extras[extra] if requirement.startswith("onnxruntime")]
        assert "onnxruntime>=1.16,<1.24; python_version < '3.11'" in runtimes
        assert "onnxruntime>=1.16,<2; python_version >= '3.11'" in runtimes


def test_ci_runs_locked_checks_and_clean_wheel_smoke():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert '["3.10", "3.11", "3.12"]' in workflow
    for command in (
        "uv sync --locked",
        "ruff check",
        "ruff format --check",
        "pytest",
        "pre-commit run --all-files",
        "uv build",
        "twine check",
        "garbage --version",
    ):
        assert command in workflow


def test_make_clean_removes_src_egg_info_and_is_phony():
    makefile = (ROOT / "Makefile").read_text()

    assert re.search(r"^\.PHONY:.*\bclean\b", makefile, re.MULTILINE)
    assert "src/*.egg-info" in makefile


def test_relative_markdown_links_resolve():
    link_pattern = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
    broken = []
    for document in (ROOT / "README.md", ROOT / "README.zh-CN.md", *sorted((ROOT / "docs").rglob("*.md"))):
        for raw_target in link_pattern.findall(document.read_text()):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "/")):
                continue
            if not (document.parent / target).exists():
                broken.append(f"{document.relative_to(ROOT)} -> {raw_target}")
    assert not broken, "broken relative Markdown links:\n" + "\n".join(broken)


def test_beginner_code_tour_and_examples_are_packaged_in_source_tree():
    assert (ROOT / "docs/code-tour.md").is_file()
    assert (ROOT / "docs/code-tour.zh-CN.md").is_file()
    assert len(list((ROOT / "examples").glob("[0-9][0-9]_*.py"))) == 5


def test_root_readmes_offer_one_beginner_workflow():
    english = (ROOT / "README.md").read_text()
    chinese = (ROOT / "README.zh-CN.md").read_text()

    assert "## Start here" in english
    assert "## 从这里开始" in chinese
    for readme in (english, chinese):
        assert "configs/learning_minimal.yaml" in readme
        assert "garbage train" in readme
        assert "garbage evaluate" in readme
        assert "garbage predict" in readme
        assert "garbage show-config" in readme


def test_code_directories_explain_their_audience_and_role():
    for directory in ("scripts", "src", "tests"):
        for name in ("README.md", "README.zh-CN.md"):
            guide = ROOT / directory / name
            assert guide.is_file(), f"missing {guide.relative_to(ROOT)}"
            assert len(guide.read_text().splitlines()) >= 10


def test_core_package_has_bilingual_local_navigation():
    package = ROOT / "src/garbage_classifier"
    for name in ("README.md", "README.zh-CN.md"):
        guide = package / name
        assert guide.is_file(), f"missing {guide.relative_to(ROOT)}"
        assert "cli.py" in guide.read_text()
        assert "training/" in guide.read_text()


def test_configuration_reference_covers_every_user_field():
    from garbage_classifier.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig

    references = [
        (ROOT / "docs/config-reference.md").read_text(),
        (ROOT / "docs/config-reference.zh-CN.md").read_text(),
    ]
    keys = {
        *(f"data.{field.name}" for field in fields(DataConfig)),
        *(f"model.{field.name}" for field in fields(ModelConfig)),
        *(f"train.{field.name}" for field in fields(TrainConfig)),
        *(field.name for field in fields(ExperimentConfig) if field.name not in {"data", "model", "train"}),
    }
    for reference in references:
        missing = sorted(key for key in keys if f"`{key}`" not in reference)
        assert not missing, f"configuration reference is missing: {missing}"
