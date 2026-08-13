import re
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
