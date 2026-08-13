# Configuration and Parameter Flow

Every training run uses one resolved `ExperimentConfig`. Parameters are not
passed as unrelated globals or copied separately into training and inference.

## Precedence

Later sources win:

```text
config.py defaults
  < YAML config
  < --set dotted override
  < dedicated CLI flags (--device, --output-dir)
```

Example:

```bash
garbage show-config --config configs/resnet50.yaml \
  --set train.lr 0.0001 \
  --set train.batch_size 16 \
  --device cpu
```

`show-config` performs the same parsing and validation as `train`; it only stops
before loading data or building a model. Use it whenever you are unsure which
value is active.

## Where parameters go

| Config section | Main consumers | Examples |
|---|---|---|
| `data` | transforms, manifest preparation, DataLoader | image size, normalization, split, workers |
| `model` | model registry | model name, pretrained weights, class count |
| `train` | Trainer and training helpers | epochs, optimizer, LR, MixUp, EMA |
| top level | command orchestration | device, output directory, run name |

The resolved config is saved as `artifacts/<run>/config.yaml`. Checkpoints also
carry model, class and preprocessing metadata for consistent inference.

## Two kinds of function parameters

- Workflow functions accept the config object: `train_from_config(cfg)`.
- Small algorithms accept explicit values: `confusion_matrix(labels, preds,
  num_classes)`.

This distinction keeps experiment settings together while leaving reusable
algorithms easy to test.
