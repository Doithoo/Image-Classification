# 小示例

[English](README.md) | [教程](../docs/tutorial/README.zh-CN.md)

每个文件隔离一个概念，除依赖 checkpoint 的示例外都可本地运行，不需要下载示例数据集。

| 示例 | 关注点 |
| --- | --- |
| `01_tensor_flow.py` | tensor、logit、loss 和梯度 |
| `02_dataset_batch.py` | dataset 与 batch 契约 |
| `03_minimal_training.py` | 最小参数更新循环 |
| `04_predict.py` | 仅依赖 checkpoint 的预测 |
| `05_export_onnx.py` | 从 checkpoint 导出 ONNX |

```bash
uv run python examples/01_tensor_flow.py --help
uv run python examples/02_dataset_batch.py --help
uv run python examples/03_minimal_training.py --help
```

示例用于解释，不是基准运行。需要比较的结果应使用已验证 manifest 和正式 CLI。
