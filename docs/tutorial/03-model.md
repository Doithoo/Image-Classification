# Models And Transfer Learning

[Previous: data](02-data.md) | [Next: training](04-training.md)

A registered model factory receives the prepared class count and replaces its
classification head. Inspect available specifications without downloading any
weights:

```bash
uv run garbage list-models
```

The minimal configuration uses a randomly initialized ResNet-18 to expose the
basic loop. For small real datasets, ImageNet initialization is usually a
better baseline:

```bash
uv run garbage train --config configs/resnet50.yaml \
  --set data.data_dir data/raw --set run_name resnet50-baseline
```

Use `model.preprocessing: model_default` when you want the registry's ImageNet
input normalization and size instead of the dataset-specific fixed values.
See [choosing models](../guides/choosing-models.md) for tradeoffs.
