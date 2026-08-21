# Dataset Format And Identity

[简体中文](dataset-format.zh-CN.md) | [Data tutorial](../tutorial/02-data.md)

`prepare-data` atomically publishes these files:

```text
dataset.yaml   schema-v2 dataset contract and combined identity
train.csv      path,label rows
valid.csv
test.csv
source.yaml    compatibility subset for older readers
summary.txt    human-readable digest summary
```

CSV paths are POSIX paths relative to `data_root`; labels are contiguous,
zero-based indexes into ordered `classes`. `dataset.yaml` records each split's
full SHA-256, per-class counts, source-image aggregate SHA-256, split policy and
an `identity` derived from all of them.

`verify-data` validates every file, label, source image byte and the combined
identity. A data modification is not automatically wrong, but it is a new
dataset contract: review it and prepare new manifests before making a metric
claim.
