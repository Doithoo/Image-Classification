# Reproducible Experiments

This guide turns the project into a small image-classification laboratory. Each
experiment changes one factor at a time, selects checkpoints on validation data,
and evaluates the selected model on the test split only once.

> **中文版：[experiments.zh-CN.md](experiments.zh-CN.md)**

## Before you run

Prepare the checked dataset and fixed manifests:

```bash
python scripts/download_data.py
garbage prepare-data --set data.data_dir data/raw --strict
```

For every run, retain:

- the resolved `config.yaml` and `metrics.csv`;
- the source archive checksum and dataset quality-patch version;
- the manifest checksum from `data/manifests/summary.txt`;
- the dependency lock, random seed and selected checkpoint.

Use validation metrics to compare settings. Run `garbage evaluate` on the test
split only after choosing the final setting.

## Experiment 1: Complete training workflow

This run establishes a reference result and exercises the full pipeline.

```bash
garbage train \
  --config configs/imbalance_none.yaml \
  --set model.name resnet50 \
  --set train.epochs 15 \
  --set train.batch_size 32 \
  --set train.lr 0.001 \
  --set train.mixup_alpha 0.2 \
  --set data.num_workers 0 \
  --set run_name exp-resnet50-mixup

garbage evaluate \
  --checkpoint artifacts/exp-resnet50-mixup/best.pt \
  --plot
```

Record accuracy, balanced accuracy, macro F1, per-class recall and the main
confusion pairs. On this imbalanced dataset, balanced accuracy and macro F1 are
more informative than accuracy alone.

## Experiment 2: Pretrained vs from scratch

Run the same MobileNetV3 configuration twice and change only pretraining:

```bash
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set model.pretrained true \
  --set train.epochs 15 --set train.mixup_alpha 0.2 \
  --set data.num_workers 0 --set run_name exp-pretrained

garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set model.pretrained false \
  --set train.epochs 15 --set train.mixup_alpha 0.2 \
  --set data.num_workers 0 --set run_name exp-scratch
```

Compare convergence speed and best validation balanced accuracy. Pretrained
weights are usually a strong starting point for small datasets, while training
from scratch is useful for understanding the value of transfer learning.

## Experiment 3: Class-imbalance strategies

The three supplied configs differ only in how they treat class imbalance:

```bash
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set run_name exp-imbalance-none
garbage train --config configs/imbalance_weighted_loss.yaml \
  --set model.name mobilenetv3_small_100 --set run_name exp-imbalance-loss
garbage train --config configs/imbalance_weighted_sampler.yaml \
  --set model.name mobilenetv3_small_100 --set run_name exp-imbalance-sampler
```

Compare validation balanced accuracy, macro F1, and `trash` precision/recall.
Weighting and oversampling often trade precision for recall; neither strategy is
automatically better for every deployment goal.

## Experiment 4: MixUp and EMA

Use one base configuration and vary one option at a time:

```bash
# Reference run
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set train.mixup_alpha 0 \
  --set train.ema false --set run_name exp-regularization-none

# MixUp
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set train.mixup_alpha 0.2 \
  --set train.ema false --set run_name exp-regularization-mixup

# EMA
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set train.mixup_alpha 0 \
  --set train.ema true --set train.ema_decay 0.99 \
  --set run_name exp-regularization-ema
```

Plot the curves with `python scripts/plot_metrics.py <metrics.csv>`. MixUp often
raises training loss while improving validation generalization. EMA needs enough
optimizer steps to catch up with the training weights, so interpret short runs
carefully.

## Experiment 5: Test-time augmentation

After selecting one checkpoint, compare plain inference and TTA:

```bash
garbage evaluate --checkpoint artifacts/<run>/best.pt
garbage evaluate --checkpoint artifacts/<run>/best.pt --tta
```

TTA averages predictions from the original and horizontally flipped images. It
costs an additional inference pass, so report both metric change and latency.

## Result table

Use this table for your own runs rather than comparing results from different
manifests or environments:

| Run | Changed factor | Best valid balanced acc | Test acc | Test balanced acc | Macro F1 | Notes |
|---|---|---:|---:|---:|---:|---|
| reference | none | | | | | |
| experiment | | | | | | |

Useful follow-up tools:

```bash
python scripts/plot_metrics.py artifacts/<run>/metrics.csv
garbage explain --checkpoint artifacts/<run>/best.pt \
  --image data/raw/paper/paper1.jpg --output gradcam.png
```
