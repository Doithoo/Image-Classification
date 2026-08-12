# Experiments (Reproducible Log)

> Purpose: this is a learning project — these experiments demonstrate the full
> workflow (train → evaluate → plot → compare), and every command is directly
> reproducible. Hardware: Apple Silicon (MPS).
>
> **Ground rule**: the truth about every experiment is `artifacts/<run>/config.yaml`
> (including the `pretrained` flag), never memory — an early version of this doc
> mislabeled pretrained fine-tuning as "from scratch", and only the config file
> exposed the mistake. That is exactly why checkpoints are self-contained.
>
> **中文版：[experiments.zh-CN.md](experiments.zh-CN.md)**

## Experiment 1: ResNet50 pretrained fine-tune (15 epochs, MixUp)

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
```

**Test-set results** (`garbage evaluate --checkpoint artifacts/exp-resnet50-mixup/best.pt --plot`):

| Class | Precision | Recall | F1 | support |
|---|---:|---:|---:|---:|
| cardboard | 0.951 | 0.951 | 0.951 | 41 |
| glass | 0.904 | 0.922 | 0.913 | 51 |
| metal | 0.857 | 0.878 | 0.867 | 41 |
| paper | 0.950 | 0.950 | 0.950 | 60 |
| plastic | 0.915 | 0.878 | 0.896 | 49 |
| trash | 0.714 | 0.714 | 0.714 | 14 |
| **accuracy** | | | **0.906** | 256 |
| **balanced acc** | | | **0.882** | |
| **macro F1** | | | **0.882** | |

With TTA (`--tta`) accuracy rises to 0.922 / balanced 0.905.

## Experiment 2: MobileNetV3-Small from-scratch vs pretrained (15 epochs, MixUp)

| Setup | valid bal | test acc | test balanced | params |
|---|---:|---:|---:|---:|
| from scratch (`pretrained: false`) | 0.492 | 0.551 | 0.479 | 1.5M |
| ImageNet pretrained fine-tune | 0.697 | 0.781 | 0.750 | 1.5M |

**Lesson**: pretraining is a huge win on small models (+27 balanced-acc points) —
a small model has limited capacity and learns poorly from scratch. But in
experiment 1 the large resnet50 trained from scratch with strong augmentation
matched fine-tuning — see experiment 4's lesson about domain shift. **Conclusion:
try pretrained fine-tuning first, then compare against from-scratch with strong
augmentation; let the experiments decide.**

## Experiment 3: class-imbalance strategies (MobileNetV3-Small pretrained, 15 epochs, no MixUp)

The three configs differ only in the rebalancing strategy
(`configs/imbalance_*.yaml`); everything else is identical. The best checkpoint
is chosen by validation balanced accuracy.

| Strategy | valid bal | test acc | test balanced | trash P | trash R | trash F1 |
|---|---:|---:|---:|---:|---:|---:|
| none (baseline) | 0.818 | **0.832** | **0.815** | 0.833 | 0.714 | **0.769** |
| inverse loss | 0.836 | 0.812 | 0.790 | 0.600 | 0.643 | 0.621 |
| weighted sampler | **0.837** | 0.816 | 0.812 | 0.647 | **0.786** | 0.710 |

**Lessons (counter-intuitive but real)**:
1. Rebalancing wins on the **validation** set but the baseline wins on the
   **test** set — model selection on validation does not guarantee transfer to
   the test set, and the smaller the sample the larger the variance.
2. Inverse loss pushes trash precision down to 0.60 and its F1 drops from 0.769
   to 0.621 — rebalancing is a precision↔recall trade-off, not free lunch.
3. The weighted sampler does raise trash recall (0.714 → 0.786) at a small cost
   to overall accuracy.

## Experiment 4: EMA on/off (MobileNetV3-Small pretrained, 15 epochs, MixUp)

| Config | valid bal | test acc | test balanced | trash F1 |
|---|---:|---:|---:|---:|
| no EMA | 0.697 | **0.781** | **0.750** | **0.696** |
| EMA decay=0.99 | 0.695 | 0.777 | 0.731 | 0.571 |
| EMA decay=0.999 | 0.397 | 0.465 | 0.417 | 0.333 |

**Lessons (three real bugs we hit, all genuine)**:
1. **EMA shadow weights MUST include BN running_mean/var.** The initial
   implementation averaged weights only and kept the fast model's BN stats —
   validation predictions collapsed to "always guess one class" (balanced acc
   frozen at 1/6). When weights change fast (fine-tuning), the BN mismatch
   saturates activations. Fixed by averaging the running stats too (this is what
   timm's `ModelEmaV2` does).
2. **The decay time constant is 1/(1−decay) steps**: 0.999 → 1000 steps,
   0.99 → 100 steps. A 15-epoch run has only ~945 steps, so the 0.999 shadow
   barely follows the training and scores far below the fast model.
3. **EMA is designed for long training**: it smooths late-training weight
   jitter. On short runs it at best ties, at worst hurts; to see its benefit you
   need tens-to-hundreds of epochs (or a lower decay).

## Experiment 5: TTA (test-time augmentation)

On `exp-resnet50-mixup`, test set:

| Method | test acc | test balanced |
|---|---:|---:|
| plain inference | 0.906 | 0.882 |
| TTA (horizontal-flip averaging) | **0.922** | **0.905** |

TTA costs one extra inference pass and buys 1.6 accuracy points — averaging the
probabilities over augmented views lowers the variance of a single decision.
`evaluate --tta` / `predict --tta` are always available.

## Tool examples

```bash
# Grad-CAM: see "where the model looked" (first step of error analysis)
garbage explain --checkpoint artifacts/exp-resnet50-mixup/best.pt \
  --image data/raw/paper/paper1.jpg --output gradcam.png

# 1-batch sanity check before a long run (data/model/device OK?)
garbage train --config configs/resnet50.yaml --dry-run

# params / FLOPs for all models
garbage bench

# plot loss curves
python scripts/plot_metrics.py artifacts/<run>/metrics.csv
```

## Artifact convention

Every run leaves in `artifacts/<run>/`: `config.yaml` (full config, the source
of truth), `metrics.csv` (per-epoch metrics), `best.pt` / `last.pt`
(self-contained checkpoints), plus `predictions.csv` / `errors.csv` /
`confusion_matrix.png` after evaluation, and optionally `loss_curve.png`.
