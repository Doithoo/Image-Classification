# Garbage Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/<owner>/Image-Classification/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

Reproducible image classification for 6 garbage categories (cardboard, glass,
metal, paper, plastic, trash) with PyTorch — config-driven training, full
evaluation metrics, self-contained checkpoints and lightweight inference.

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

# 2. data: generate portable manifests from class-folder data
garbage prepare-data --set data.data_dir <path/to/class-folders>

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
- The dataset is not stored in git; see `scripts/download_data.py`
  (release hosting is pending — until then point `--data-dir` at a local copy).

## Project structure

```
src/garbage_classifier/
  config.py          typed config (YAML + dotted overrides)
  cli.py             train / evaluate / predict / prepare-data
  data/              manifests, dataset, transforms
  models/            registry + timm zoo (+ legacy shim)
  training/          Trainer, checkpoint save/load
  evaluation/        metrics, classification report, confusion matrix
  inference/         Predictor (checkpoint-driven)
  utils/             seeding, device, git revision, logging
configs/             example experiment YAMLs
tests/               unit + CPU smoke tests
docs/baseline.md     frozen pre-refactor state
```

## Reproducibility guarantees

- [ ] fresh env: `pip install -e .` + one CLI command → CPU smoke run
- [ ] any complete experiment replayable from `artifacts/<run>/` alone
- [ ] test split never used for tuning; all comparisons traceable (config+seed+weights)
- [ ] inference never requires manually repeating classes / normalization / model name
- [ ] CI green on lint + unit tests + minimal training flow

## Roadmap

- [x] Baseline freeze & engineering foundation (package, config, CLI, tests, CI)
- [x] Portable manifests & reproducible split
- [x] Training/evaluation/inference core
- [ ] Imbalance ablations (weighted loss / sampler vs none)
- [ ] Legacy model migration (registry shim + consistency check)
- [ ] ONNX export & demo app
- [ ] Dataset release (GitHub Release / HuggingFace) + v1.0

## Contributing

See the roadmap above; PRs welcome. Run `ruff check .` and `pytest` locally,
and use conventional commit messages (`feat:`, `fix:`, `docs:`, …).

## License

[MIT](LICENSE)
