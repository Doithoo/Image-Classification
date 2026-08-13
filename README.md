# Garbage Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/Image-Classification/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**Read this in: [简体中文](README.zh-CN.md)**

A reproducible image-classification project for 6 garbage categories
(cardboard, glass, metal, paper, plastic, trash) built with PyTorch —
config-driven training, honest evaluation metrics, self-contained checkpoints,
and lightweight inference.

> **Purpose: a learning project.** The code is written to be *correct, clear and
> explained* — every non-trivial module carries "why" comments, and the docs form
> a guided path from zero to training your own image classifiers.
>
> If you are a beginner, start with the
> **[Learning Path](docs/learning-path.md)** — it tells you exactly what to read,
> run and practice at each step, ending with you training and evaluating your own
> models from scratch.

## Table of contents

- [Features](#features)
- [Quick start](#quick-start)
- [Learning path (for beginners)](#learning-path-for-beginners)
- [Configuration](#configuration)
- [Model zoo](#model-zoo-timm-backbones)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Class-imbalance ablations](#class-imbalance-ablations)
- [Reproducibility guarantees](#reproducibility-guarantees)
- [Documentation](#documentation)
- [License](#license)

## Features

- **Config-driven experiments** — models, hyper-parameters and data settings live
  in YAML; switch models with one flag, never edit source code.
- **Portable data pipeline** — CSV manifests with platform-independent relative
  paths; reproducible stratified split (fixed seed), bad/duplicate image
  validation, data summary + checksum.
- **Self-contained inference checkpoints** — weights, class map, preprocessing
  stats and model config travel together, so prediction needs no training config.
- **Class-imbalance aware** — best checkpoint selected by `balanced_accuracy` /
  `macro_f1`; full per-class precision/recall/F1 report, confusion matrix,
  prediction CSV and error-sample list.
- **Professional training loop** — AMP, resume, early stopping, warmup + cosine
  LR, label smoothing, MixUp/CutMix, EMA, gradient clipping, full RNG seeding,
  per-epoch CSV metrics.
- **Model zoo** — maintained timm/torchvision backbones and educational
  hand-written implementations behind a single registry.
- **Interpretability & deployment** — Grad-CAM heatmaps (`garbage explain`),
  TTA, ONNX export, Gradio demo.
- **Quality gates** — ruff lint/format, behavioral pytest coverage, GitHub Actions CI
  (Python 3.10–3.12), pre-commit hooks.

## Quick start

```bash
# 1. create and activate an environment (Python >= 3.10)
uv venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# 2. download, checksum, audit-patch, then build portable manifests
python scripts/download_data.py            # -> data/raw/
garbage prepare-data --set data.data_dir data/raw

# 3. run a lightweight, stable-name MobileNet experiment
garbage train --config configs/resnet50.yaml \
  --set model.name mobilenetv3_small_100 \
  --set train.epochs 3 --set train.batch_size 16 \
  --set data.num_workers 0 --set run_name quickstart-mobilenet

# 4. evaluate the best checkpoint on the held-out test split
garbage evaluate --checkpoint artifacts/quickstart-mobilenet/best.pt --plot

# 5. predict a single image or a whole folder
garbage predict --checkpoint artifacts/quickstart-mobilenet/best.pt \
  --image data/raw/paper/paper1.jpg --top-k 3
garbage predict --checkpoint artifacts/quickstart-mobilenet/best.pt --image <folder>

# 6. explore: Grad-CAM heatmap and model size table
garbage explain --checkpoint artifacts/quickstart-mobilenet/best.pt \
  --image data/raw/paper/paper1.jpg
garbage bench
```

ONNX and the web demo use optional dependencies:

```bash
uv pip install -e ".[onnx]"
garbage export-onnx --checkpoint artifacts/quickstart-mobilenet/best.pt --output model.onnx

uv pip install -e ".[demo]"
garbage demo --checkpoint artifacts/quickstart-mobilenet/best.pt
```

One-line CPU smoke run (sanity check before a long run):

```bash
garbage train --config configs/resnet50.yaml --set model.name mobilenetv3_small_100 \
  --set train.epochs 1 --set train.batch_size 8 --set data.num_workers 0 \
  --set device cpu --set run_name quickstart-smoke
# or verify the whole pipeline on a single batch:
garbage train --config configs/resnet50.yaml --dry-run
```

## Learning path (for beginners)

**Don't know where to start? Follow the [Learning Path](docs/learning-path.md).**
It is a 7-stage curriculum that takes you from *"what is image classification?"*
to *"I trained and evaluated my own models, and I can explain every number on
the report"* — each stage lists what to read, what to run, and practice exercises
with expected observations.

| Stage | Topic | Key takeaway |
|---|---|---|
| 1 | The problem & the data | class imbalance, train/val/test discipline |
| 2 | Data pipeline | portable manifests, transforms, why split by seed |
| 3 | First model | timm backbones, model registry |
| 4 | Training loop | loss, optimizer, LR schedule, warmup, MixUp, EMA |
| 5 | Evaluation | accuracy vs balanced accuracy, F1, confusion matrix |
| 6 | Inference & explanation | predict, Grad-CAM, TTA, ONNX |
| 7 | Experiments & reproducibility | config.yaml as the source of truth |

Companion docs: [`docs/how-it-works.md`](docs/how-it-works.md) explains the data
flow and every technique in the training loop; [`docs/experiments.md`](docs/experiments.md)
contains runnable experiments and the lessons they illustrate.

## Configuration

Every experiment is a YAML file with three sections (`data`, `model`, `train`);
any subset overrides the defaults (see `src/garbage_classifier/config.py`). CLI
flags can override any value with dotted keys:

```bash
garbage train --config configs/resnet50.yaml \
  --set train.lr 1e-4 \
  --set model.name swin_tiny_patch4_window7_224 \
  --set train.epochs 80
```

The resolved config is saved in `artifacts/<run>/config.yaml`. A checkpoint is
enough for standalone inference. Reproducing a training result additionally
  requires the exact manifests, source data version (including quality patch), and
dependency lock used by the run; an artifact directory alone is not sufficient.

## Model zoo (timm backbones)

| Registry key | Params* | Notes |
|---|---:|---|
| `resnet18/34/50/101/152` | 11.7M–60.2M | classic baselines |
| `convnext_tiny/small` | 28.6M–50.2M | strong accuracy/compute |
| `mobilenetv3_small_100/large_100` | 2.5M–5.5M | mobile-friendly |
| `efficientnet_b0/b3`, `efficientnetv2_s` | 5.3M–21.5M | FLOPs-optimized |
| `swin_tiny_patch4_window7_224` | 28.3M | transformer |
| `vit_base_patch16_224` | 86.6M | transformer |
| `regnetx_002/004` | 1.8M–5.2M | light |
| `tv_resnet50` | 23.5M | torchvision parity baseline |

\* ImageNet-pretrained parameter counts (timm). Run `garbage bench` for the full
table with FLOPs; `available_models()` lists all registry keys.

## Dataset

- The upstream v1 archive contains 2,527 images. The downloader verifies that
  archive, then applies quality patch `v1.0-audit.1`, removing three reviewed bad
  labels/duplicates before the 80/10/10 stratified split.
- **Imbalance**: `trash` is only 5.4% of training data while `paper` is 23.5% —
  accuracy alone is misleading, so evaluation always reports balanced accuracy,
  macro/weighted F1 and per-class metrics.
- Hosted as a GitHub Release asset (not in git):
  `garbage-classification.tar.gz` (39.7 MB, SHA-256 `be0b99fc…ea1b6`).
  Fetch it with `python scripts/download_data.py` (verifies the checksum).
- The original source is the public "Garbage classification" dataset on Kaggle.

## Project structure

```
Image-Classification/
├── src/garbage_classifier/
│   ├── cli.py            CLI presentation layer (arg parsing + dispatch)
│   ├── config.py         typed config (YAML + dotted overrides)
│   ├── data/             manifests, dataset, transforms, prepare-data logic
│   ├── models/           registry + timm zoo + educational implementations
│   │   └── legacy_models/   hand-written reference models (lazy-loaded)
│   ├── training/         Trainer, checkpoints, train command logic
│   ├── evaluation/       metrics, reports, evaluate command logic
│   ├── inference/        Predictor, Grad-CAM, ONNX export, demo
│   └── utils/            seeding, device, git revision, logging
├── configs/              example experiment YAMLs
├── scripts/              download_data.py, plot_metrics.py
├── tests/                unit + CPU end-to-end smoke tests
├── docs/                 learning path, how-it-works, experiment log
├── data/                 (gitignored) downloaded dataset + generated manifests
└── artifacts/            (gitignored) per-run: config.yaml, metrics.csv, best.pt, plots
```

## Class-imbalance ablations

`trash` is only 5.4% of the training data, so three comparable strategies are
provided (best checkpoint is selected by `balanced_accuracy` in these configs):

| Config | Strategy |
|---|---|
| `configs/imbalance_none.yaml` | plain CE loss, uniform sampling (baseline) |
| `configs/imbalance_weighted_loss.yaml` | inverse-frequency class weights in the loss |
| `configs/imbalance_weighted_sampler.yaml` | `WeightedRandomSampler` (oversample rare classes) |

The example results and complete commands are documented in
[`docs/experiments.md`](docs/experiments.md#experiment-3-class-imbalance-strategies).

## Reproducibility guarantees

- [x] fresh env: `pip install -e .` + one CLI command → CPU smoke run
- [x] checkpoints carry everything needed for standalone inference
- [ ] full experiment replay requires archived manifests, source-data quality-patch
  version, config, weights and a dependency lock
- [x] inference never requires manually repeating classes / normalization / model name
- [x] CI green on lint + unit tests + minimal training flow

## Documentation

| Doc | Language | Purpose |
|---|---|---|
| [`docs/learning-path.md`](docs/learning-path.md) | EN | 7-stage curriculum for beginners |
| [`docs/learning-path.zh-CN.md`](docs/learning-path.zh-CN.md) | 中文 | 初学者 0→1 学习路线 |
| [`docs/tutorial/`](docs/tutorial/README.zh-CN.md) | 中文 | hands-on tutorials: devices, data, models, training strategy |
| [`docs/how-it-works.md`](docs/how-it-works.md) | EN | data flow + why every technique exists ([中文](docs/how-it-works.zh-CN.md)) |
| [`docs/experiments.md`](docs/experiments.md) | EN | reproducible experiment log + lessons ([中文](docs/experiments.zh-CN.md)) |

## License

[MIT](LICENSE)
