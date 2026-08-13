# 代码导览：初学者应该先读什么

完整工程代码不会像课堂伪代码一样短。本导览的目标不是让你一次读完所有文件，而是
给出一条从简单示例进入真实实现的路线。

## 阅读顺序

| 步骤 | 先运行 | 再阅读 | 重点理解 |
|---|---|---|---|
| 1 | `python examples/01_tensor_flow.py` | `data/transforms.py` | 图片、Tensor、logits、概率 |
| 2 | `python examples/02_dataset_batch.py` | `data/dataset.py` | Dataset 与 DataLoader 的契约 |
| 3 | `python examples/03_minimal_training.py` | `training/trainer.py::_run_epoch` | 前向、loss、反向、更新 |
| 4 | `garbage train --config configs/learning_minimal.yaml` | `training/train.py` | 如何组装数据、模型和 Trainer |
| 5 | `python examples/04_predict.py --help` | `inference/predictor.py` | 如何恢复预处理和权重 |
| 6 | `python examples/05_export_onnx.py --help` | `inference/export.py` | checkpoint 如何变成部署文件 |

## 最小训练循环

先读 `examples/03_minimal_training.py`。核心只有：

```python
optimizer.zero_grad(set_to_none=True)
outputs = model(images)
loss = loss_fn(outputs, labels)
loss.backward()
optimizer.step()
```

完整 `Trainer` 仍然遵循这个顺序。AMP 改变反向和更新的执行方式；MixUp 改变输入与
标签；EMA 保存另一组平滑权重；早停与 checkpoint 包裹在 epoch 外围。它们是在基础
循环上逐层添加能力，不是另一套训练方法。

## 第一遍可以跳过什么

- `config.py` 中 dataclass 之后的代码：严格解析与校验基础设施。
- `data/manifest.py` 的私有辅助函数：重复组分配和元数据处理。
- `training/checkpoint.py`、`ema.py`、`mixup.py`：完成基础训练后再看。
- `models/legacy_models/`：手写网络结构参考，不是日常选模型入口。
- `cli.py`：命令行表现层和错误处理，不是深度学习核心。

当你遇到对应问题时再回来读这些文件。按“数据如何流动”阅读，比按文件名字母顺序
阅读更容易建立整体认识。
