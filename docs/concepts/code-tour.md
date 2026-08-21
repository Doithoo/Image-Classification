# Code Tour

This tour answers one question: **which code should a beginner read first?** The
production package is complete by design, so do not start by reading every file.

For directory roles and parameter precedence, see
[Configuration and Parameter Flow](configuration-flow.md) and the READMEs inside
`scripts/`, `src/` and `tests/`. When you enter the installed package, keep its
[local package map](../../src/garbage_classifier/README.md) open beside the code.

> **中文版：[code-tour.zh-CN.md](code-tour.zh-CN.md)**

## Reading order

| Step | Run | Then read | Learn |
|---|---|---|---|
| 1 | `python examples/01_tensor_flow.py` | `data/transforms.py` | image, tensor, logits, probabilities |
| 2 | `python examples/02_dataset_batch.py` | `data/dataset.py` | Dataset and DataLoader contract |
| 3 | `python examples/03_minimal_training.py` | `training/trainer.py::_run_epoch` | forward, loss, backward, update |
| 4 | `garbage train --config configs/learning_minimal.yaml` | `training/train.py` | assemble data, model and trainer |
| 5 | `python examples/04_predict.py --help` | `inference/predictor.py` | restore preprocessing and weights |
| 6 | `python examples/05_export_onnx.py --help` | `inference/export.py` | turn a checkpoint into a deployment artifact |

## The smallest useful loop

Read `examples/03_minimal_training.py` first. Its core is:

```python
optimizer.zero_grad(set_to_none=True)
outputs = model(images)
loss = loss_fn(outputs, labels)
loss.backward()
optimizer.step()
```

The full `Trainer` preserves that order. AMP changes how backward/update are
performed; MixUp changes images and targets; EMA keeps a second set of weights;
early stopping and checkpoints happen around epochs. These are additions to the
same loop, not a different training process.

## What to skip on the first pass

- `config.py` below the dataclasses: strict parsing and validation infrastructure.
- `data/manifest.py` private helpers: duplicate-group optimization and metadata.
- `training/checkpoint.py`, `ema.py`, `mixup.py`: read after one basic run.
- `cli.py`: command-line presentation and error handling, not deep-learning logic.

Return to these files when the corresponding question arises. Reading by data
flow is more useful than reading the package alphabetically.
