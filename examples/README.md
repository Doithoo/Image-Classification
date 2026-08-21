# Small Examples

[简体中文](README.zh-CN.md) | [Tutorial](../docs/tutorial/README.md)

Each file isolates one idea and runs locally without downloading the example
dataset, except checkpoint-dependent examples.

| Example | Focus |
| --- | --- |
| `01_tensor_flow.py` | tensors, logits, loss and gradients |
| `02_dataset_batch.py` | dataset and batch contract |
| `03_minimal_training.py` | minimal parameter-update loop |
| `04_predict.py` | checkpoint-only prediction |
| `05_export_onnx.py` | ONNX export from a checkpoint |

```bash
uv run python examples/01_tensor_flow.py --help
uv run python examples/02_dataset_batch.py --help
uv run python examples/03_minimal_training.py --help
```

Examples are explanatory, not benchmark runs. Use a verified manifest and the
normal CLI for results you intend to compare.
