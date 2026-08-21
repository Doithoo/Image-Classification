# PyTorch Image Classification Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/pytorch-image-classification-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**Read this in: [Simplified Chinese](README.zh-CN.md)**

A reproducible, beginner-oriented PyTorch image-classification project. The
included example classifies cardboard, glass, metal, paper, plastic, and trash;
the same workflow supports any one-folder-per-class dataset.

## Start Here

Install the project, obtain the audited example data, then run the shortest
complete path:

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py
uv run garbage prepare-data --set data.data_dir data/raw
uv run garbage verify-data --set data.data_dir data/raw
uv run garbage train --config configs/learning_minimal.yaml --dry-run --set data.data_dir data/raw
```

A dry run verifies data loading, model construction, forward pass, loss and
backward pass. It leaves no run directory or checkpoint. Start an isolated
training run when the check succeeds:

```bash
uv run garbage train --config configs/learning_minimal.yaml \
  --set data.data_dir data/raw --set run_name first-run
uv run garbage evaluate --checkpoint artifacts/first-run/best.pt \
  --set data.data_dir data/raw --plot
uv run garbage predict --checkpoint artifacts/first-run/best.pt \
  --image data/raw/paper/paper1.jpg --top-k 3
```

Each run records its resolved configuration, dataset identity, environment,
metrics and safe, versioned checkpoints. Evaluation evidence is written under
`artifacts/<run>/evaluation/<split>/` and is never silently overwritten.

![Reference ResNet50 training curves](docs/assets/reference-resnet50-curves.png)

The [recorded reference run](docs/recorded-run/README.md) reached 93.33% test
accuracy on its fixed six-class split. It is a scoped reproducibility record,
not a universal benchmark.

## Documentation

Use the [documentation home](docs/README.md) to choose a path:

- [Tutorial](docs/tutorial/README.md): first environment, data, model, training and inference workflow.
- [Concepts](docs/concepts/classification-flow.md): classification flow, code tour and configuration flow.
- [Guides](docs/guides/using-your-data.md): own data, experiments, model choice and troubleshooting.
- [Reference](docs/reference/config-reference.md): every config field, dataset identity, metrics and checkpoints.
- [Recorded run](docs/recorded-run/README.md): machine-readable evidence for the published result.

The [configuration directory](configs/README.md) and [examples directory](examples/README.md)
explain the runnable files without requiring users to infer their purpose from names.

## Commands

```text
prepare-data  create versioned train/valid/test manifests
verify-data   verify manifests and source image identity
inspect-data  show dataset distribution and identity
show-config   inspect resolved settings and their sources
train         train or safely resume an isolated run
evaluate      publish metrics, predictions and error evidence
predict       predict files or directories with optional JSON evidence
list-models   inspect registered models without constructing them
export-onnx   export and numerically verify an ONNX graph
explain       produce a Grad-CAM overlay for supported CNNs
```

## Development

```bash
uv sync --locked --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv build
uv run twine check dist/*
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing code or documentation.
The project is released under the [MIT License](LICENSE).
