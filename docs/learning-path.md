# Image Classification Learning Path

This path is for learners who have met Python, tensors, loss, and gradients but
have not completed a full image-classification project. Reading and running it
in order takes about **8–12 hours**. The first pass is about connecting data,
training, evaluation, and inference; optional training techniques can wait.

If logits, loss, or gradients are still unfamiliar, start with the Chinese
[minimum deep-learning concepts](tutorial/00-basics.zh-CN.md). Environment and
device details are covered in the Chinese
[environment guide](tutorial/01-environment.zh-CN.md).

## Before starting

Create the environment and inspect the resolved configuration:

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
garbage --version
garbage show-config --config configs/learning_minimal.yaml
```

`show-config` prints the settings that actually take effect. It is more
reliable than describing an experiment from memory.

## Run the workflow once

Download the data, make a fixed split, and check one batch end to end:

```bash
python scripts/download_data.py
garbage prepare-data --set data.data_dir data/raw
garbage train --config configs/learning_minimal.yaml --dry-run
```

`dry-run OK` means images load, the model builds, and forward and backward
passes complete. It checks the pipeline; it does not measure model quality.

Run two epochs next:

```bash
garbage train --config configs/learning_minimal.yaml \
  --set train.epochs 2 --set run_name first-minimal-run
```

Open `artifacts/first-minimal-run/config.yaml` and `metrics.csv`. The first
records the exact setup, while the second shows what changed between epochs.

## Inspect the data

Before interpreting a model, inspect what it is being asked to classify:

```bash
python scripts/preview_dataset.py --data-dir data/raw
cat data/manifests/summary.txt
```

The preview exposes misplaced or surprising images. `summary.txt` records the
classes, split sizes, seed, and manifest checksum.

Notice that `trash` has fewer examples than the other classes. Accuracy alone
can hide that weakness. Validation data compares configurations; test data is
reserved for the selected result. A fixed seed keeps those comparisons on the
same split.

Read the data sections of [How It Works](how-it-works.md) and
`src/garbage_classifier/data/manifest.py` for the implementation.

## Follow an image into a batch

An image reaches the training loop through the manifest, Dataset, transforms,
and DataLoader. Read these files in order:

```text
src/garbage_classifier/data/manifest.py
src/garbage_classifier/data/dataset.py
src/garbage_classifier/data/transforms.py
```

Inspect one sample directly:

```bash
python -c "
from garbage_classifier.data import ImageClassificationDataset
from garbage_classifier.data.transforms import build_train_transform
from garbage_classifier.config import load_config
cfg = load_config('configs/learning_minimal.yaml')
ds = ImageClassificationDataset('data/manifests/train.csv',
      transform=build_train_transform(cfg.data))
image, label = ds[0]
print(image.shape, label)
"
```

The result is a `[3, 224, 224]` tensor and an integer label. Training
transforms are random; validation transforms are deterministic. The checkpoint
carries preprocessing values so inference does not silently use different
normalization.

## See how the model predicts

Start with `src/garbage_classifier/models/registry.py` and `models/zoo.py`.
`model.name` chooses the backbone, while the manifest class list determines the
classification head size.

```bash
garbage bench
garbage show-config --config configs/learning_minimal.yaml
```

Parameter count is useful context, not a quality ranking. Pretrained models
often reach a useful result faster on small datasets; scratch training makes
the effect of initialization and data size easier to see.

The Chinese [model guide](tutorial/03-model.zh-CN.md) goes deeper.

## Read the training loop

Run the minimal example before opening the full trainer:

```bash
python examples/03_minimal_training.py
```

```text
for each batch:
  outputs = model(images)
  loss = loss_fn(outputs, labels)
  loss.backward()
  optimizer.step()

for each epoch:
  evaluate on validation data
  update best.pt and last.pt
  append metrics.csv
```

The production implementation is in
`src/garbage_classifier/training/trainer.py`. Read the basic loop in
`_run_epoch` first, then see how `fit` adds validation, checkpoints, learning
rate scheduling, and early stopping. The minimal config disables MixUp, EMA,
and AMP so optional branches do not obscure the first reading.

Falling training loss means the model is fitting training data. Stalled or
worsening validation metrics may instead point to overfitting, data problems,
or an unsuitable configuration.

## Judge the result

Evaluate the best checkpoint on the held-out test split:

```bash
garbage evaluate \
  --checkpoint artifacts/first-minimal-run/best.pt \
  --plot
```

Evaluation prints accuracy, balanced accuracy, macro F1, per-class metrics,
and a confusion matrix. It also writes `predictions.csv` and `errors.csv`.

Compare accuracy with balanced accuracy first. A large gap can indicate that
rare classes are being ignored. Then inspect per-class recall and the confusion
matrix before opening individual errors. The pattern helps distinguish label
or image-quality problems from a weak model.

Metric code lives in `src/garbage_classifier/evaluation/metrics.py`; the
interpretation is expanded in [How It Works](how-it-works.md).

## Use the checkpoint

A checkpoint contains weights, model identity, class names, preprocessing, and
the resolved training config. Prediction therefore needs only the checkpoint
and an image:

```bash
garbage predict \
  --checkpoint artifacts/first-minimal-run/best.pt \
  --image data/raw/paper/paper1.jpg \
  --top-k 3
```

For a wrong prediction, Grad-CAM can show where the model concentrated:

```bash
garbage explain \
  --checkpoint artifacts/first-minimal-run/best.pt \
  --image data/raw/paper/paper1.jpg \
  --output artifacts/first-minimal-run/gradcam.png
```

Grad-CAM is a clue, not a causal explanation. Attention on the background or
an unrelated object is a reason to inspect the data and transforms more
closely.

## Make a useful comparison

After the complete workflow runs, choose one question and change only one
factor. Keep the manifest, seed, model, epochs, and preprocessing fixed unless
one of them is the variable being tested.

For example, compare pretrained and scratch initialization:

```bash
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 \
  --set model.pretrained true \
  --set train.epochs 15 \
  --set run_name compare-pretrained

garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 \
  --set model.pretrained false \
  --set train.epochs 15 \
  --set run_name compare-scratch
```

Predict which run will converge faster, then compare the best validation
balanced accuracy and macro F1. The result describes this dataset, setup, and
environment; it is not a universal statement about transfer learning.

More runnable comparisons are in [Experiments](experiments.md). The Chinese
[training guide](tutorial/04-training.zh-CN.md) helps diagnose unusual curves.

## Where to go next

- Use the [Code Tour](code-tour.md) to follow boundaries between modules.
- Read [Configuration Flow](configuration-flow.md) to understand overrides.
- Explore TTA, Grad-CAM, ONNX, or Gradio in the Chinese
  [inference and deployment guide](tutorial/05-inference-deployment.zh-CN.md).
- Look up individual settings in the
  [Configuration Reference](config-reference.md).

There is no need to read everything in one sitting. Returning to the relevant
section when a real question appears is usually more useful than reading the
entire source tree in sequence.
