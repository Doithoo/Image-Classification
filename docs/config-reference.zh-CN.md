# 配置参数参考

运行 `garbage show-config` 可以查看真正会生效的参数。下列默认值来自 `config.py`，
YAML 和命令行可以覆盖它们。

## 数据参数

| 参数 | 默认值 | 含义与建议 |
|---|---|---|
| `data.data_dir` | `data` | `prepare-data` 扫描的类别目录根路径；下载数据通常在 `data/raw`，需显式传入。 |
| `data.manifest_dir` | `data/manifests` | 切分 CSV 与 `source.yaml` 的位置；训练从这里读取。 |
| `data.classes` | 六类垃圾名称 | 期望类别名；训练时以生成的 manifest 元数据为准。 |
| `data.image_size` | `224` | 最终方形输入尺寸；越大越耗内存和算力，也影响 ONNX 输入。 |
| `data.resize_size` | `256` | 裁剪前的缩放尺寸，必须不小于 `image_size`。 |
| `data.normalize_mean` | `[0.673, 0.639, 0.604]` | RGB 均值，训练/评测/推理共用并保存进 checkpoint。 |
| `data.normalize_std` | `[0.208, 0.209, 0.231]` | RGB 正标准差，必须和 mean 来自同一数据来源。 |
| `data.split_ratios` | `[0.8, 0.1, 0.1]` | train/valid/test 比例，三个非负数之和必须为 1。 |
| `data.seed` | `666` | 控制 manifest 固定切分，不控制模型训练。 |
| `data.num_workers` | `4` | 并行加载进程数；调试或受限系统使用 `0`。 |
| `data.pin_memory` | `true` | 加快 CPU 到 CUDA 的传输，只在 CUDA 设备实际启用。 |
| `data.aug` | `basic` | `basic` 或 `randaug`；先用基础增强，过拟合时再加强。 |

## 模型参数

| 参数 | 默认值 | 含义与建议 |
|---|---|---|
| `model.name` | `resnet50` | 模型注册键；运行 `garbage bench`。初学可先用 `resnet18` 或 MobileNetV3。 |
| `model.num_classes` | `null` | 自动从 manifest 推导；只有想断言类别数量时才显式填写。 |
| `model.pretrained` | `true` | 加载 ImageNet 权重；小数据集通常建议开启，从零训练练习时关闭。 |

## 训练参数

| 参数 | 默认值 | 含义与建议 |
|---|---|---|
| `train.epochs` | `60` | 最大训练轮数；练习用 2-5，早停可能提前结束。 |
| `train.batch_size` | `32` | 每次更新的图片数；OOM 时减半，并重新考虑学习率。 |
| `train.lr` | `0.001` | 学习率；预训练微调常从 `0.0001`-`0.0005` 开始。 |
| `train.weight_decay` | `0.0001` | 参数正则化强度，`0` 表示关闭。 |
| `train.momentum` | `0.9` | SGD 动量，AdamW/Lion 会忽略；范围 `[0, 1)`。 |
| `train.optimizer` | `adamw` | `adamw`、`sgd` 或可选的 `lion`；初学默认 AdamW。 |
| `train.scheduler` | `cosine` | `cosine`、`step` 或 `none`，每个 epoch 后调整学习率。 |
| `train.warmup_epochs` | `5` | 线性预热轮数；最小教学配置使用 `0`。 |
| `train.label_smoothing` | `0.1` | 标签平滑，范围 `[0, 1)`；学习基础 loss 时设为 `0`。 |
| `train.mixup_alpha` | `0.0` | 正数开启 MixUp；与 CutMix 同时开启时优先 MixUp。 |
| `train.cutmix_alpha` | `0.0` | MixUp 关闭时，正数开启 CutMix。 |
| `train.grad_clip` | `1.0` | 梯度范数上限，`0` 表示关闭裁剪。 |
| `train.amp` | `true` | CUDA/MPS 混合精度；排查数值问题时可关闭。 |
| `train.early_stop_patience` | `15` | 连续多少轮无提升后停止；`0` 会在第一次未提升时停止。 |
| `train.best_metric` | `macro_f1` | 选择 `best.pt` 的指标，必须是评测输出中存在的指标。 |
| `train.ema` | `false` | 保存平滑部署权重；理解基础循环后再开启。 |
| `train.ema_decay` | `0.999` | EMA 保留率，范围 `[0, 1)`；越高越需要长训练。 |
| `train.class_weight` | `none` | 类别不平衡 loss 加权：`none`、`inverse`、`effective`。 |
| `train.sampler` | `none` | `none` 或 `weighted`，改变训练 batch 中的样本频率。 |
| `train.seed` | `42` | 控制模型初始化、增强与训练随机性。 |

## 顶层参数

| 参数 | 默认值 | 含义与建议 |
|---|---|---|
| `device` | `auto` | `auto`、`cpu`、`cuda` 或 `mps`；专用 `--device` 优先级最高。 |
| `output_dir` | `artifacts` | 运行产物父目录；专用 `--output-dir` 优先级最高。 |
| `run_name` | `null` | 运行目录名；省略时使用模型名加时间戳。 |
| `log_level` | `info` | `debug`、`info`、`warning`、`error` 或 `critical`。 |

## 参数之间的重要关系

- `data.resize_size >= data.image_size`；修改图片尺寸会同时影响内存、算力、预处理和
  ONNX 输入形状。
- `data.seed` 控制数据切分，`train.seed` 控制训练随机性，两者不是一回事。
- 减小 `train.batch_size` 可以解决 OOM；batch 大幅变化时通常也要按比例调整 lr。
- `train.warmup_epochs` 应小于 `train.epochs`，余弦训练才有实际意义；校验器只要求非负。
- MixUp/CutMix 会产生软标签；两个 alpha 都为正时使用 MixUp。
- loss 加权和 weighted sampler 可以叠加，但可能过度矫正，建议一次只比较一个变量。
- `model.num_classes: null` 最安全，由 manifest 提供真实类别数量。
- checkpoint 推理会恢复模型和预处理元数据，不需要手动重复训练配置。

## 仅属于命令行的参数

以下是操作参数，不属于实验配置：`--resume`、`--dry-run`、`--checkpoint`、
`--split`、`--plot`、`--tta`、`--top-k`、`--image`、`--output`、`--opset`、
`--no-verify`、`--class-idx`、`--alpha`、`--share`、`--strict`、`--legacy`。
运行 `garbage <命令> --help` 查看它们的取值和作用范围。
