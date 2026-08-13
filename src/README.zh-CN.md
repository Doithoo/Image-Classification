# 正式源码导览

[English](README.md)

`src/garbage_classifier/` 是安装后真正运行的应用。初学者不要按文件名字母顺序通读，
先跟随[代码导览](../docs/code-tour.zh-CN.md)。

```text
命令行参数和 YAML
        ↓
cli.py → config.py → ExperimentConfig
        ↓
data/ → models/ → training/ → checkpoint
                              ↓
                    evaluation/ + inference/
```

| 区域 | 职责 | 第一次先读 |
|---|---|---|
| `data/` | manifest、Dataset、变换 | `dataset.py`，再读 `transforms.py` |
| `models/` | 模型名称和创建 | `registry.py`，再读 `zoo.py` |
| `training/` | 组装并训练实验 | `train.py`，再读 `trainer.py` |
| `evaluation/` | 指标和报告 | `metrics.py` |
| `inference/` | 预测、解释和导出 | `predictor.py` |

`cli.py` 是命令行表现层；`legacy_models/` 是手写结构参考，属于选读内容。
自动生成的 `*.egg-info/` 和 `__pycache__/` 可以忽略。

参数按照“默认值 < YAML < `--set` < 专用参数”的顺序覆盖。训练前可运行：

```bash
garbage show-config --config configs/resnet50.yaml \
  --set train.lr 0.0001 --device cpu
```
