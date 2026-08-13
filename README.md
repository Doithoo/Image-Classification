# Garbage Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/Image-Classification/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**Read this in: [简体中文](README.zh-CN.md)**

A small, reproducible PyTorch project for learning image classification from
data preparation to deployment. The example task has six classes:
cardboard, glass, metal, paper, plastic, and trash.

This repository is designed to be read and run in order. You do not need to
understand every file before your first experiment.

## Start here

Use this path for your first successful run:

    README -> learning path -> minimal config -> train -> evaluate -> predict

1. Read the [Learning Path](docs/learning-path.md). It explains what to learn
   at each stage and points to a runnable exercise.
2. Install the project and run the [minimal configuration](configs/learning_minimal.yaml).
3. Train for two or three epochs, inspect the evaluation report, then predict
   one image.
4. Only after that, read the [Code Tour](docs/code-tour.md) and change one
   configuration value at a time.

The project assumes basic Python. If tensors, logits, loss, or gradients are
new to you, start with the [minimum deep-learning concepts](docs/tutorial/00-basics.zh-CN.md).

## What is included

- Reproducible train, validation, and test manifests with data-quality checks.
- YAML experiments with visible defaults and command-line overrides.
- Maintained CNN and Transformer backbones with optional pretrained weights.
- A readable training loop with checkpoints, resume, AMP, early stopping,
  MixUp/CutMix, EMA, and class-imbalance strategies.
- Per-class precision, recall, F1, balanced accuracy, confusion matrices, and
  error samples instead of accuracy alone.
- Prediction, TTA, Grad-CAM, ONNX export, and a Gradio demo.

## Install

Python 3.10, 3.11, or 3.12 is supported. The commands below use uv, but a
normal virtual environment and pip install -e ".[dev]" work as well.

    uv venv
    source .venv/bin/activate              # Windows: .venv\Scripts\activate
    uv pip install -e ".[dev]"

Check the command before downloading data:

    garbage --help
    garbage show-config --config configs/learning_minimal.yaml

## First experiment

Download the public dataset, build portable manifests, and run a short CPU or
GPU experiment. The download script verifies the archive before it is used;
downloaded images and training artifacts remain local and are not committed.

    python scripts/download_data.py
    garbage prepare-data --set data.data_dir data/raw
    garbage train --config configs/learning_minimal.yaml --dry-run

    garbage train --config configs/learning_minimal.yaml \
      --set train.epochs 2 --set run_name first-run

The run writes its configuration, metrics, and checkpoints to
artifacts/first-run/. Before a longer run, preview the final merged settings:

    garbage show-config --config configs/learning_minimal.yaml \
      --set train.epochs 5 --device cpu

## Use the checkpoint

Evaluation uses the held-out test split. Prediction restores the class names
and preprocessing values from the checkpoint, so you do not repeat training
settings by hand.

    garbage evaluate --checkpoint artifacts/first-run/best.pt --plot
    garbage predict --checkpoint artifacts/first-run/best.pt \
      --image data/raw/paper/paper1.jpg --top-k 3

After the basic workflow, try the optional tools:

    garbage explain --checkpoint artifacts/first-run/best.pt \
      --image data/raw/paper/paper1.jpg
    garbage bench

    uv pip install -e ".[onnx]"
    garbage export-onnx --checkpoint artifacts/first-run/best.pt --output model.onnx

    uv pip install -e ".[demo]"
    garbage demo --checkpoint artifacts/first-run/best.pt

## What you will learn

| Stage | Question | Main result |
|---|---|---|
| 1. Data | What is being classified and how should it be split? | Audited manifests and class counts |
| 2. Pipeline | How does an image become a batch of tensors? | Dataset, transforms, and DataLoader |
| 3. Model | What are a backbone, a head, and a registry? | A configurable pretrained classifier |
| 4. Training | How do loss, gradients, and learning rate change weights? | A reproducible training loop |
| 5. Evaluation | Why is accuracy not enough? | Per-class metrics and confusion matrix |
| 6. Inference | How do I use a trained model? | Single-image, batch, and TTA prediction |
| 7. Deployment | How can another program run it? | Grad-CAM, ONNX, and Gradio |

Follow the complete [learning path](docs/learning-path.md), or use the Chinese
[hands-on tutorials](docs/tutorial/README.zh-CN.md) for detailed explanations.

## Configuration and models

Experiments are described in YAML. Values are resolved in this order:

    dataclass defaults < YAML file < --set overrides < dedicated CLI flags

Read the [Configuration Reference](docs/config-reference.md) for every field,
default, and relationship. The [minimal config](configs/learning_minimal.yaml)
keeps advanced techniques off so the first training loop is easy to inspect.

The model registry contains maintained timm and torchvision backbones:

- resnet18, resnet34, resnet50, resnet101, resnet152
- mobilenetv3_small_100, mobilenetv3_large_100
- efficientnet_b0, efficientnet_b3, efficientnetv2_s
- convnext_tiny, convnext_small, regnetx_002, regnetx_004
- swin_tiny_patch4_window7_224, vit_base_patch16_224, tv_resnet50

Run garbage bench to compare parameter counts and, when available, FLOPs.

## Repository map

    examples/                    five small programs to run and read first
    configs/                     ready-to-run experiment settings
    docs/                        learning path, tutorials, and code tour
    scripts/                     download, preview, and plotting utilities
    src/garbage_classifier/      installed application and reusable package
    tests/                       behavior checks and advanced reference examples
    data/                        local dataset and manifests (gitignored)
    artifacts/                   local runs and checkpoints (gitignored)

When you are learning, use examples/ -> docs/ -> src/. When contributing, read
[CONTRIBUTING.md](CONTRIBUTING.md), then explore tests/, pyproject.toml, and CI.
Each code directory has its own guide: [scripts/README.md](scripts/README.md),
[src/README.md](src/README.md), and [tests/README.md](tests/README.md).

## Documentation

- [Learning Path](docs/learning-path.md) · [中文学习路线](docs/learning-path.zh-CN.md)
- [Code Tour](docs/code-tour.md) · [中文代码导览](docs/code-tour.zh-CN.md)
- [Configuration Reference](docs/config-reference.md) · [中文参数参考](docs/config-reference.zh-CN.md)
- [How It Works](docs/how-it-works.md) · [中文原理说明](docs/how-it-works.zh-CN.md)
- [Experiments](docs/experiments.md) · [中文实验记录](docs/experiments.zh-CN.md)
- [Hands-on tutorials](docs/tutorial/README.zh-CN.md)

## License

[MIT](LICENSE)
