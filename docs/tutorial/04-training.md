# Training And Checkpoints

[Previous: models](03-model.md) | [Next: evaluation](05-evaluation-and-deployment.md)

Run a one-batch check before longer work:

```bash
uv run garbage train --config configs/learning_minimal.yaml \
  --set data.data_dir data/raw --dry-run
```

The check leaves no run artifacts. Then create a named run:

```bash
uv run garbage train --config configs/learning_minimal.yaml \
  --set data.data_dir data/raw --set run_name baseline
```

A run has a resolved `config.yaml`, atomic `metrics.csv`, `run.yaml`, `best.pt`
and `last.pt`. New checkpoint schema v2 binds model, preprocessing, ordered
classes and `manifest_identity`. A run directory cannot be reused accidentally.
Resume only with `last.pt` in the same directory:

```bash
uv run garbage train --config artifacts/baseline/config.yaml \
  --resume artifacts/baseline/last.pt
```

Read the [checkpoint schema](../reference/checkpoint-schema.md) for the exact
contract and compatibility limits.
