# Reproducible Experiments

These optional comparisons help answer a specific question without changing
several settings at once. Select checkpoints with validation data and leave the
test split untouched until the final setting has been chosen.

> **中文版：[experiments.zh-CN.md](experiments.zh-CN.md)**

## Before you run

Prepare the checked dataset and fixed manifests:

```bash
python scripts/download_data.py
garbage prepare-data --set data.data_dir data/raw --strict
```

The generated files already preserve most of the useful context:

- the resolved `config.yaml` and `metrics.csv`;
- the source archive checksum and dataset quality-patch version;
- the manifest checksum from `data/manifests/summary.txt`;
- the dependency lock, random seed and selected checkpoint.

Use validation metrics to compare settings. Run `garbage evaluate` on the test
split only after choosing the final setting.

## A worked way to reason about a comparison

The [verified ResNet50 run](reference-run.md) is a useful control because its
manifest, configuration, environment, and output are all visible. Suppose the
question is: **does MixUp improve validation performance for this setup?**

- Keep the manifest, seed, ResNet50 model, 15 epochs, optimizer, scheduler, and
  preprocessing unchanged.
- Change one setting: `train.mixup_alpha` from `0.2` to `0.0`.
- The control reached `0.8976` validation balanced accuracy and `0.8858` macro
  F1 at epoch 15. The comparison run must use validation metrics from its own
  `metrics.csv`, not the test result.
- In the control's held-out errors, `trash` had 14 examples and the lowest
  recall (`0.7143`). Check whether its validation behavior and error samples
  change rather than relying only on an aggregate score.
- One run per setting can suggest what happened here, but it cannot establish
  a general rule about MixUp. Seed variation and another dataset may change the
  conclusion.

This order keeps the question, controlled settings, changed setting,
observation, and limitation connected. The same reasoning works for the short
comparisons below.

## Optional comparison 1: Complete training workflow

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

Compare accuracy, balanced accuracy, macro F1, per-class recall, and the main
confusion pairs. On this imbalanced dataset, balanced accuracy and macro F1 are
more informative than accuracy alone.

## Optional comparison 2: Pretrained vs from scratch

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

## Optional comparison 3: Class-imbalance strategies

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

## Optional comparison 4: MixUp and EMA

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

## Optional comparison 5: Test-time augmentation

After selecting one checkpoint, compare plain inference and TTA:

```bash
garbage evaluate --checkpoint artifacts/<run>/best.pt
garbage evaluate --checkpoint artifacts/<run>/best.pt --tta
```

TTA averages predictions from the original and horizontally flipped images. It
costs an additional inference pass, so inspect both the metric change and
latency before deciding whether it helps.

Useful follow-up tools:

```bash
python scripts/plot_metrics.py artifacts/<run>/metrics.csv
garbage explain --checkpoint artifacts/<run>/best.pt \
  --image data/raw/paper/paper1.jpg --output gradcam.png
```
