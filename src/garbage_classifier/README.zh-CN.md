# 核心代码包

[English](README.md)

这里是安装后真正运行的 `garbage_classifier` 包。建议先运行
[`examples/`](../../examples/) 并阅读[代码导览](../../docs/concepts/code-tour.zh-CN.md)，
然后把本页作为正式实现的地图。

## 训练命令调用链

```text
garbage train
  → cli.main()                    解析命令行参数
  → cmd_train()                   生成唯一的 ExperimentConfig
  → config.load_config()          默认值 + YAML + 覆盖参数
  → training.train_from_config()  创建数据、DataLoader 和模型
  → models.create_model()         从注册表返回 nn.Module
  → training.Trainer.fit()        epoch 循环和验证
  → Trainer._run_epoch()          前向、loss、反向、更新
  → checkpoint.save_checkpoint()  保存 best.pt 和 last.pt
```

## 包内地图

| 模块 | 输入 | 输出 | 什么时候读 |
|---|---|---|---|
| `config.py` | YAML 和覆盖参数 | `ExperimentConfig` | 理解配置时 |
| `data/` | 类别目录/manifest | Tensor batch | 学习数据管道时 |
| `models/` | 模型名、类别数 | `nn.Module` | 选择和创建模型时 |
| `training/` | 配置、模型、loader | checkpoint、指标 | 学习优化过程时 |
| `evaluation/` | checkpoint、数据划分 | 报告、错误样本、图 | 评测模型时 |
| `inference/` | checkpoint、图片 | 概率、热力图、ONNX | 使用训练模型时 |
| `utils/` | 小型运行参数 | 设备、种子、日志工具 | 被引用时再读 |
| `cli.py` | 终端参数 | 调用上述模块 | 学习应用如何串联时 |

## 分层阅读

1. `data/dataset.py`、`data/transforms.py`、`models/registry.py`。
2. `training/train.py`，再读 `training/trainer.py` 中标记的 `BASIC LOOP`。
3. `evaluation/metrics.py`、`inference/predictor.py`。
4. `mixup.py`、`ema.py`、`checkpoint.py` 和 manifest 私有辅助函数。

所有用户配置参数见[配置参数参考](../../docs/reference/config-reference.zh-CN.md)。
