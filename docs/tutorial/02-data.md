# Data And Manifests

[Previous: environment](01-environment.md) | [Next: models](03-model.md)

The source layout is one folder per class:

```text
data/raw/
  cardboard/example.jpg
  glass/example.jpg
```

Create a fixed split and inspect its identity:

```bash
uv run garbage prepare-data --set data.data_dir data/raw
uv run garbage verify-data --set data.data_dir data/raw
uv run garbage inspect-data --set data.data_dir data/raw
```

Preparation validates readable images, detects content-identical duplicates,
prevents identical bytes from crossing splits and atomically writes
`dataset.yaml`, all CSVs and `summary.txt`. See the
[dataset format reference](../reference/dataset-format.md) before adapting a
custom dataset.
