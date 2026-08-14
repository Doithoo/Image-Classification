# Verified ResNet50 Reference Run

## Why this run exists

This run provides a result that can be inspected rather than a score to copy.
It connects the configuration, learning curves, selected checkpoint, test
metrics, and class-level errors from one completed run. Its purpose is to make
the evaluation workflow concrete for a learner.

## Dataset and fixed split

The audited example dataset contains 2,524 images in six classes. The strict
split created 2,017 training images, 252 validation images, and **255 held-out
test images** with seed `666` and ratios `0.8/0.1/0.1`. The training manifest
checksum recorded in `data/manifests/summary.txt` was
`4ad59108d1d0d216`.

That checksum identifies the manifest used for this result. A different data
download, split, or manifest can produce a different result even when the
model configuration is unchanged.

## Exact configuration

The run used [the published ResNet50 configuration](../configs/reference_resnet50.yaml):
15 epochs, batch size 32, AdamW, cosine scheduling with five warmup epochs,
label smoothing `0.1`, MixUp `0.2`, RandAugment, and a pretrained ResNet50.
Balanced accuracy selected `best.pt`.

## Recorded environment

The generated `run.yaml` recorded:

- started at `2026-08-14T07:33:20.319748+00:00`;
- elapsed time `1636.6` seconds (about 27 minutes 17 seconds);
- Python `3.11.15` and PyTorch `2.13.0`;
- `macOS-26.5.2-arm64-arm-64bit`, device `mps`, accelerator `arm64`;
- Git revision `dc2b22a`.

The revision is useful context: the later default-data-path fix does not
change the manifest or model inputs used by this run.

## Validation result

Epoch 15 had the best validation balanced accuracy, `0.8976`. Its validation
accuracy was `0.8929`, macro F1 was `0.8858`, and validation loss was `0.6352`.
The curves show a generally falling validation loss with a few normal
epoch-to-epoch reversals rather than a perfectly smooth improvement.

![Training and validation curves](assets/reference-resnet50-curves.png)

## Held-out test result

Evaluation of `best.pt` produced `evaluation.json` with these aggregate values:

| Metric | Value |
| --- | ---: |
| Accuracy | `0.9333` |
| Balanced accuracy | `0.9079` |
| Macro F1 | `0.9176` |
| Weighted F1 | `0.9325` |

The distinction matters: validation metrics selected the checkpoint; the 255
test images were used afterward for the reported result.

## What the classes reveal

Glass had the highest per-class F1 (`0.9709`, 51 images). The rare `trash`
class had the lowest recall (`0.7143`) and F1 (`0.8000`): 10 of its 14 images
were correct, while four were confused once each with cardboard, metal,
paper, and plastic. With only 14 test images, one mistake moves trash recall by
more than seven percentage points, so this class needs sample-level inspection
rather than a confident general claim.

![Test-set confusion matrix](assets/reference-resnet50-confusion.png)

## What this result means

This is evidence that the repository can reproduce a complete ResNet50
workflow and retain enough metadata to inspect the result. It is a reference
for this dataset manifest, configuration, software revision, and machine. It
is not a universal ResNet50 benchmark, evidence that every class is solved, or
a promise that another device or dataset will produce the same score.

## Reproduce it

```bash
python scripts/download_data.py
garbage prepare-data --strict --set data.data_dir data/raw
garbage train --config configs/reference_resnet50.yaml
garbage evaluate --config configs/reference_resnet50.yaml \
  --checkpoint artifacts/reference-resnet50/best.pt \
  --output-dir artifacts/reference-resnet50/evaluation --plot
python scripts/plot_metrics.py artifacts/reference-resnet50/metrics.csv \
  --out artifacts/reference-resnet50/reference-resnet50-curves.png
```

Compare the new `run.yaml`, `metrics.csv`, and `evaluation.json` with the
values above. Small numerical differences can still occur across hardware and
library implementations even with the same seeds.
