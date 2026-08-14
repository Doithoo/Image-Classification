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


def test_root_readmes_present_the_learning_project_identity():
    english = (ROOT / "README.md").read_text()
    chinese = (ROOT / "README.zh-CN.md").read_text()

    assert english.startswith("# PyTorch Image Classification Lab\n")
    assert chinese.startswith("# PyTorch 图像分类实践\n")
    assert "8–12 hours" in english
    assert "8–12 小时" in chinese


def test_tracked_user_links_use_the_current_repository_name():
    old_slug = "Doithoo/" + "Image-Classification"
    documents = [
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "scripts/download_data.py",
        ROOT / "docs/tutorial/01-environment.zh-CN.md",
    ]

    findings = [str(path.relative_to(ROOT)) for path in documents if old_slug in path.read_text()]
    assert not findings, f"stale repository links found: {findings}"


def test_readme_images_resolve():
    image_pattern = re.compile(r"!\[[^]]*\]\(([^)]+)\)")
    broken = []
    for document in (ROOT / "README.md", ROOT / "README.zh-CN.md"):
        for raw_target in image_pattern.findall(document.read_text()):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>").split("#", 1)[0]
            if target and "://" not in target and not (document.parent / target).is_file():
                broken.append(f"{document.name} -> {raw_target}")
    assert not broken, "broken README images:\n" + "\n".join(broken)


def test_chinese_learning_docs_avoid_classroom_language():
    documents = [
        ROOT / "README.zh-CN.md",
        ROOT / "docs/learning-path.zh-CN.md",
        *sorted((ROOT / "docs/tutorial").glob("*.zh-CN.md")),
    ]
    forbidden = ("我们", "毕业项目", "课程表", "通关", "闯关")
    findings = []
    for document in documents:
        text = document.read_text()
        for phrase in forbidden:
            if phrase in text:
                findings.append(f"{document.relative_to(ROOT)}: {phrase}")
    assert not findings, "classroom-style language found:\n" + "\n".join(findings)


def test_own_data_guides_use_isolated_manifests_and_the_minimal_config():
    for name in ("using-your-data.md", "using-your-data.zh-CN.md"):
        guide = (ROOT / "docs" / name).read_text()
        assert "my-dataset" in guide
        assert "data/my-manifests" in guide
        assert "configs/learning_minimal.yaml" in guide
        assert "--strict" in guide
        assert "model.num_classes" in guide

    assert "docs/using-your-data.md" in (ROOT / "README.md").read_text()
    assert "docs/using-your-data.zh-CN.md" in (ROOT / "README.zh-CN.md").read_text()
    assert "using-your-data.md" in (ROOT / "docs/learning-path.md").read_text()
    assert "using-your-data.zh-CN.md" in (ROOT / "docs/learning-path.zh-CN.md").read_text()
    assert "../using-your-data.zh-CN.md" in (ROOT / "docs/tutorial/README.zh-CN.md").read_text()
    assert "../using-your-data.zh-CN.md" in (ROOT / "docs/tutorial/02-data.zh-CN.md").read_text()


def test_reference_run_has_config_docs_and_visual_evidence():
    expected = [
        ROOT / "configs/reference_resnet50.yaml",
        ROOT / "docs/reference-run.md",
        ROOT / "docs/reference-run.zh-CN.md",
        ROOT / "docs/assets/reference-resnet50-curves.png",
        ROOT / "docs/assets/reference-resnet50-confusion.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.is_file()]
    assert not missing, f"missing reference evidence: {missing}"

    for name in ("reference-run.md", "reference-run.zh-CN.md"):
        text = (ROOT / "docs" / name).read_text()
        for fact in ("run.yaml", "evaluation.json", "255", "ResNet50"):
            assert fact in text

    assert "docs/reference-run.md" in (ROOT / "README.md").read_text()
    assert "docs/reference-run.zh-CN.md" in (ROOT / "README.zh-CN.md").read_text()


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
