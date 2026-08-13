# Contributing

[简体中文](CONTRIBUTING.zh-CN.md)

## Setup

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
pre-commit install
```

Create a focused branch, add tests for behavior changes, and keep documentation
commands runnable from the repository root. Do not commit datasets, artifacts,
credentials, or generated model files.

## Validate

```bash
ruff check .
ruff format --check .
pytest
```

For data or experiment changes, record the source archive checksum, dataset
quality-patch version, manifest checksum, resolved config, dependency lock and exact
command. Select models on validation data; do not use the test split for tuning.

Use English ASCII [Conventional Commits](https://www.conventionalcommits.org/),
for example `fix(data): Reject conflicting duplicate labels`. Keep the subject
at 72 characters or fewer, capitalize the imperative description, and omit the
trailing period.

Open a bug report for broken behavior, a learning issue for unclear educational
material, or a pull request using the supplied templates.
