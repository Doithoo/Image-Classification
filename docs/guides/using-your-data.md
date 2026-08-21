# Use Your Own Image Dataset

This guide applies the same workflow to another folder-based classification
dataset. It assumes that the project is already installed and that each image
has one class label.

## Arrange the folders

Put images directly inside one folder per class:

```text
my-dataset/
├── apples/
│   ├── apple-001.jpg
│   └── apple-002.jpg
├── bananas/
│   └── banana-001.jpg
└── oranges/
    └── orange-001.jpg
```

Supported suffixes are `.jpg`, `.jpeg`, and `.png`. The scanner reads only the
files directly inside each class folder; deeper nested folders are ignored.

## Preview before splitting

Look at the images before starting a long run:

```bash
python scripts/preview_dataset.py --data-dir my-dataset
```

Open the generated preview and check that folder names match the visible
content. This catches misplaced images and unclear class boundaries before they
become training errors.

## Prepare separate manifests

Keep manifests for this dataset separate from the garbage example:

```bash
garbage prepare-data --strict \
  --set data.data_dir my-dataset \
  --set data.manifest_dir data/my-manifests
```

`--strict` stops on duplicate image content. If it reports a duplicate, inspect
the named files before deciding which copy to keep.

Read the two metadata files next:

```bash
cat data/my-manifests/summary.txt
cat data/my-manifests/source.yaml
```

Check the discovered classes, per-class counts, split sizes, and data root.
Class names and `model.num_classes` come from `source.yaml`; do not hard-code a
class count in the model configuration.

## Check one batch

Use the minimal configuration first:

```bash
garbage train --config configs/learning_minimal.yaml --dry-run \
  --set data.data_dir my-dataset \
  --set data.manifest_dir data/my-manifests \
  --set run_name my-data-check
```

`dry-run OK` confirms that the manifests, transforms, labels, model output, and
backward pass agree. It does not measure whether the model is useful.

## Run a short baseline

Keep the first run short and avoid changing several settings at once:

```bash
garbage train --config configs/learning_minimal.yaml \
  --set data.data_dir my-dataset \
  --set data.manifest_dir data/my-manifests \
  --set train.epochs 3 \
  --set run_name my-data-baseline
```

Three epochs provide an initial curve and checkpoint. They are a pipeline and
data check, not a competitive result.

Evaluate with the same data and manifest locations:

```bash
garbage evaluate --checkpoint artifacts/my-data-baseline/best.pt \
  --set data.data_dir my-dataset \
  --set data.manifest_dir data/my-manifests \
  --plot
```

Compare accuracy with balanced accuracy and macro F1, then inspect per-class
recall, the confusion matrix, and `errors.csv`. Those observations are a better
starting point than immediately replacing the model.

## Common problems

| Symptom | Likely cause | First check |
|---|---|---|
| No images found | Images are nested too deeply or use another suffix | Confirm `my-dataset/<class>/<image>` |
| Unreadable image error | Corrupt or incomplete file | Open the reported image separately |
| Duplicate error in strict mode | Identical content appears more than once | Compare the reported paths and labels |
| A split is missing a class | Too few images in that class | Add data before tuning the model |
| Accuracy looks acceptable but macro F1 is weak | Class imbalance or one ignored class | Inspect class counts and per-class recall |
| Training is good while validation is poor | Overfitting or a split mismatch | Check previews, duplicates, and augmentation |

Once the baseline is understood, change one factor at a time. The
[experiment guide](experiments.md) shows how to keep comparisons interpretable.
