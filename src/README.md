# Source Package Guide

[简体中文](README.zh-CN.md)

`src/garbage_classifier/` is the installed application. Beginners should follow
the [Code Tour](../docs/code-tour.md) instead of reading it alphabetically. Its
[local package guide](garbage_classifier/README.md) maps inputs, outputs and the
complete training call flow.

## Module map

```text
CLI arguments and YAML
        ↓
cli.py → config.py → ExperimentConfig
        ↓
data/ → models/ → training/
                    ↓
              checkpoint (.pt)
                    ↓
          evaluation/ + inference/
```

| Area | Responsibility | First file to read |
|---|---|---|
| `data/` | Manifests, Dataset, transforms | `dataset.py`, then `transforms.py` |
| `models/` | Model names and construction | `registry.py`, then `zoo.py` |
| `training/` | Build and train an experiment | `train.py`, then `trainer.py` |
| `evaluation/` | Metrics and reports | `metrics.py` |
| `inference/` | Prediction, explanation and export | `predictor.py` |
| `utils/` | Device, seed, logging helpers | Read when referenced |

`cli.py` is a presentation layer. It does not contain model-training logic.
`legacy_models/` contains hand-written architecture references and is optional
reading. Generated `*.egg-info/` and `__pycache__/` directories can be ignored.

## Parameter flow

Configuration-driven commands use one `ExperimentConfig` object:

```text
dataclass defaults in config.py
        ↓ overridden by
configs/<experiment>.yaml
        ↓ overridden by
--set section.key value
        ↓ overridden by
dedicated flags such as --device / --output-dir
        ↓
cli._resolve_cfg()
        ↓
train_from_config(cfg) → Trainer(..., cfg, ...)
```

Inspect the final values before running work:

```bash
garbage show-config --config configs/resnet50.yaml \
  --set train.lr 0.0001 --device cpu
```

Use the [Configuration Reference](../docs/config-reference.md) when you need the
meaning, default and relationships of individual fields.

Data preprocessing is stored in the same config and copied into checkpoints, so
training, evaluation and inference do not maintain separate hidden parameters.
