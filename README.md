# Garbage Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/<owner>/Image-Classification/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

Reproducible image classification for 6 garbage categories (cardboard, glass,
metal, paper, plastic, trash) with PyTorch — config-driven training, full
evaluation metrics, self-contained checkpoints and lightweight inference.

> **定位：学习练习项目**。代码追求正确、清晰、带"为什么"注释，适合作为
> 图像分类入门到进阶的学习脚手架。建议从
> [`docs/how-it-works.md`](docs/how-it-works.md) 读起，用
> [`docs/experiments.md`](docs/experiments.md) 里的命令亲手复现实验。

This is the refactored successor of the original script-based project (see
[`docs/baseline.md`](docs/baseline.md) for the frozen pre-refactor state).

## Features

- **Config-driven experiments** — models, hyper-parameters and data settings live
  in YAML; switch models with one flag, never edit source code.
- **Portable data pipeline** — CSV manifests with platform-independent relative
  paths replace the old Windows-only `train.txt` files; reproducible stratified
  split (fixed seed), bad/duplicate image validation, data summary + checksum.
- **Self-contained checkpoints** — weights, optimizer/scheduler state, class map,
  preprocessing stats, full config, git revision and RNG state; inference needs
  only the checkpoint, no config drift.
- **Class-imbalance aware** — best checkpoint selected by `macro_f1` /
  `balanced_accuracy` (default), full per-class precision/recall/F1 report,
  confusion matrix, prediction CSV and error-sample list.
- **Professional training loop** — AMP, resume, early stopping, cosine/step LR,
  label smoothing, gradient clipping, full RNG seeding, per-epoch CSV metrics.
- **Model zoo** — timm maintained backbones (ResNet/ConvNeXt/MobileNet/EfficientNet/
  Swin/ViT/…) behind a single registry; legacy hand-written models migrate into
  the registry in a later phase.
- **Quality gates** — ruff lint/format, pytest suite (unit + CPU smoke), GitHub
  Actions CI, pre-commit hooks.

## Quick start

```bash
# 1. install (Python >= 3.10)
uv pip install -e ".[dev]"          # or: pip install -e ".[dev]"

# 2. data: download the dataset (GitHub Release, SHA-256 verified) and generate manifests
python scripts/download_data.py          # -> data/raw/
garbage prepare-data --set data.data_dir data/raw

# 3. train (config-driven; see configs/)
garbage train --config configs/resnet50.yaml

# 4. evaluate the best checkpoint on the held-out test split
garbage evaluate --checkpoint artifacts/resnet50-<timestamp>/best.pt

# 5. predict a single image / a folder
garbage predict --checkpoint artifacts/resnet50-<timestamp>/best.pt --image img.jpg --top-k 3
garbage predict --checkpoint artifacts/resnet50-<timestamp>/best.pt --image <folder>
```

One-line CPU smoke run (tiny, for CI or sanity):

```bash
garbage train --config configs/resnet50.yaml --set train.epochs 1 --set device cpu
```

## Configuration

Every experiment is a YAML file with three sections; any subset overrides the
defaults (see `src/garbage_classifier/config.py`). CLI flags can override any
value with dotted keys:

```bash
garbage train --config configs/resnet50.yaml \
  --set train.lr 1e-4 --set model.name swin_tiny_patch4_window7_224 --set train.epochs 80
```

The fully-resolved config is saved to every run directory
(`artifacts/<run>/config.yaml`), so any experiment can be replayed from its
artifact alone.

## Model zoo (timm backbones)

| Registry key | Params* | Notes |
|---|---:|---|
| `resnet18/34/50/101/152` | 11.7M–60.2M | classic baselines |
| `convnext_tiny/small` | 28.6M–50.2M | strong accuracy/compute |
| `mobilenetv3_small_100/large_100` | 2.5M–5.5M | mobile-friendly |
| `efficientnet_b0/b3`, `efficientnetv2_s` | 5.3M–21.5M | FLOPs-optimized |
| `swin_tiny_patch4_window7_224` | 28.3M | transformer |
| `vit_base_patch16_224` | 86.6M | transformer |
| `regnetx_002/004`, `shufflenet_v2_x0_5` | 1.8M–5.2M | light |
| `tv_resnet50` | 23.5M | torchvision parity baseline |

\* ImageNet-pretrained parameter counts (timm). Run `garbage train --help` for
usage; `available_models()` lists all registry keys.

## Dataset

- 2,527 images (512×384), 6 classes; split 80/10/10 (seed 666, stratified,
  matching the historical split exactly).
- **Imbalance**: `trash` is only 5.4% of training data while `paper` is 23.5% —
  accuracy alone is misleading, so evaluation always reports balanced accuracy,
  macro/weighted F1 and per-class metrics.
- Hosted as a GitHub Release asset (not in git):
  `garbage-classification.tar.gz` (39.7 MB, SHA-256
  `be0b99fc…ea1b6`). Fetch it with:

  ```bash
  python scripts/download_data.py        # verifies checksum, extracts to data/raw/
  ```

  If the original Kaggle source is ever needed for attribution, it is the
  "Garbage classification" dataset (2,527 images, 6 classes).

## Project structure

```
Image-Classification/
├── src/garbage_classifier/
│   ├── cli.py            CLI presentation layer (arg parsing + dispatch)
│   ├── config.py         typed config (YAML + dotted overrides)
│   ├── data/             manifests, dataset, transforms, prepare-data logic
│   ├── models/           registry + timm zoo (+ legacy shim)
│   │   └── legacy_models/   vendored hand-written models (lazy-loaded)
│   ├── training/         Trainer, checkpoints, train command logic
│   ├── evaluation/       metrics, reports, evaluate command logic
│   ├── inference/        Predictor, Grad-CAM, ONNX export, demo
│   └── utils/            seeding, device, git revision, logging
├── configs/              example experiment YAMLs
├── scripts/              download_data.py, plot_metrics.py
├── tests/                unit + CPU end-to-end smoke tests
├── docs/                 learning guide, experiment log, baseline report
├── data/                 (gitignored) downloaded dataset + generated manifests
└── artifacts/            (gitignored) per-run: config.yaml, metrics.csv, best.pt, plots
```

## Reproducibility guarantees

- [x] fresh env: `pip install -e .` + one CLI command → CPU smoke run
- [x] any complete experiment replayable from `artifacts/<run>/` alone
- [x] test split never used for tuning; all comparisons traceable (config+seed+weights)
- [x] inference never requires manually repeating classes / normalization / model name
- [x] CI green on lint + unit tests + minimal training flow

## Class-imbalance ablations

The `trash` class is only 5.4% of training data, so three comparable strategies are
provided (best checkpoint is selected by `balanced_accuracy` in these configs):

| Config | Strategy |
|---|---|
| `configs/imbalance_none.yaml` | plain CE loss, uniform sampling (baseline) |
| `configs/imbalance_weighted_loss.yaml` | inverse-frequency class weights in the loss |
| `configs/imbalance_weighted_sampler.yaml` | `WeightedRandomSampler` (oversample rare classes) |

## Legacy models

The original 16 hand-written model families are vendored under
`garbage_classifier.models.legacy_models` and exposed as `legacy_*` registry keys
(38 entries) with lazy loading — they are kept for reproducibility of historical
experiments, but the maintained timm backends are the recommended default.

## ONNX export & demo

```bash
garbage export-onnx --checkpoint artifacts/<run>/best.pt --output model.onnx
garbage demo --checkpoint artifacts/<run>/best.pt      # Gradio UI (pip install -e '.[demo]')
```

## Contributing

See the roadmap above; PRs welcome. Run `ruff check .` and `pytest` locally,
and use conventional commit messages (`feat:`, `fix:`, `docs:`, …).

## License

[MIT](LICENSE)
