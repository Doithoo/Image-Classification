# 配置参考

[English](config-reference.md) | [配置流](../concepts/configuration-flow.zh-CN.md)

解析顺序是数据类默认值、YAML、`--set KEY VALUE`，最后是专用 CLI 参数。使用 `garbage show-config --sources` 检查最终值。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `data.data_dir` | `data/raw` | 按类别分目录的数据根；通过 identity 校验后可覆盖元数据根 |
| `data.manifest_dir` | `data/manifests` | `dataset.yaml` 和 split CSV 所在目录 |
| `data.image_size` / `data.resize_size` | `224` / `256` | 裁剪和缩放尺寸 |
| `data.normalize_mean` / `data.normalize_std` | 数据集值 | fixed 预处理使用的 RGB 归一化 |
| `data.split_ratios`, `data.seed` | `[.8,.1,.1]`, `666` | 确定性数据准备策略 |
| `data.num_workers`, `data.pin_memory` | `0`, `true` | DataLoader 运行参数 |
| `data.aug` | `basic` | `basic` 或 `randaug` 增强 |
| `data.interpolation` | `bilinear` | `bilinear` 或 `bicubic` 缩放插值 |
| `model.name` | `resnet18` | 注册模型名 |
| `model.num_classes` | `null` | 可选断言；运行时从元数据推导类别数 |
| `model.pretrained` | `false` | 请求 provider 预训练权重 |
| `model.preprocessing` | `fixed` | `fixed` 项目值或 `model_default` registry ImageNet 值 |
| `model.params` | `{}` | 显式 provider/model 关键字参数 |
| `model.factory` | `null` | 可信扩展 `module:function` factory |
| `train.epochs`, `train.batch_size`, `train.lr`, `train.weight_decay`, `train.momentum`, `train.seed` | `20`, `32`, `.001`, `.0001`, `.9`, `42` | 基础优化参数 |
| `train.optimizer`, `train.scheduler` | `adamw`, `cosine` | 优化器和调度策略 |
| `train.warmup_epochs` | `0` | 必须小于 epochs |
| `train.label_smoothing`, `train.mixup_alpha`, `train.cutmix_alpha` | `0` | 正则化；MixUp 与 CutMix 互斥 |
| `train.grad_clip`, `train.amp`, `train.ema`, `train.ema_decay` | `0`, `false`, `false`, `.999` | 可选稳定性/性能工具 |
| `train.early_stop_patience` | `5` | 无改进验证 epoch 数；`0` 禁用 early stopping |
| `train.best_metric` | `macro_f1` | 已报告验证集聚合指标之一 |
| `train.class_weight`, `train.sampler` | `none`, `none` | 类别不平衡替代策略 |
| `device` | `auto` | `auto`、`cpu`、`cuda` 或 `mps` |
| `output_dir`, `run_name` | `artifacts`, `null` | 独立运行输出位置 |
| `log_level` | `info` | 控制台日志级别 |

类别不再作为配置字段。其有序映射由准备后的数据元数据负责，并嵌入每个 checkpoint。
