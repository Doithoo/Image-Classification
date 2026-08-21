# 训练配置

[English](README.md) | [配置参考](../docs/reference/config-reference.zh-CN.md)

训练前先检查最终值：

```bash
uv run garbage show-config --config configs/learning_minimal.yaml --sources
```

| 文件 | 用途 | 权重/网络 |
| --- | --- | --- |
| `learning_minimal.yaml` | 第一次 CPU dry run 和透明基线 | 不下载预训练权重 |
| `resnet50.yaml` | 实用预训练 CNN 基线 | ImageNet 权重下载/缓存 |
| `reference_resnet50.yaml` | 参考运行设置 | ImageNet 权重下载/缓存 |
| `swin_tiny.yaml` | Transformer 对照 | ImageNet 权重下载/缓存 |
| `imbalance_none.yaml` | 不平衡基线 | ImageNet 权重下载/缓存 |
| `imbalance_weighted_loss.yaml` | 逆频率损失消融 | ImageNet 权重下载/缓存 |
| `imbalance_weighted_sampler.yaml` | 加权采样消融 | ImageNet 权重下载/缓存 |

每个独立实验都使用新的 `run_name`。测试数据仅用于最终选出的模型；候选方案通过验证集证据比较。
