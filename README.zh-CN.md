# 垃圾图像分类（Garbage Image Classification）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/Image-Classification/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**English version: [README.md](README.md)**

基于 PyTorch 的垃圾分类图像分类项目（6 类：纸板 / 玻璃 / 金属 / 纸 / 塑料 /
其他垃圾）—— 配置驱动训练、诚实全面的评测指标、自包含 checkpoint、轻量推理。

> **项目定位：学习练习项目。** 代码追求"正确、清晰、讲清为什么"—— 每个非平凡
> 模块都带教学注释，文档构成一条从零到会训练自己图像分类模型的完整路线。
>
> 如果你是初学者，**从这里开始：[学习路线](docs/learning-path.zh-CN.md)** ——
> 它告诉你每一步该读什么、跑什么、练习什么，最终你会亲手训练并评测自己的模型，
> 并能解释报告上每一个数字的含义。

## 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [学习路线（初学者必读）](#学习路线初学者必读)
- [配置系统](#配置系统)
- [模型库](#模型库timm-主干网络)
- [数据集](#数据集)
- [项目结构](#项目结构)
- [类别不平衡消融实验](#类别不平衡消融实验)
- [可复现性保证](#可复现性保证)
- [文档索引](#文档索引)
- [许可证](#许可证)

## 功能特性

- **配置驱动实验** —— 模型、超参数、数据设置全部写在 YAML 里；换模型只需改一个
  参数，永远不用改源码。
- **可移植数据管道** —— CSV manifest 使用平台无关的相对路径；固定种子的分层切分、
  坏图/重复图校验、数据摘要与校验和。
- **自包含推理 checkpoint** —— 权重、类别表、预处理参数与模型配置一起保存，预测时
  不需要训练配置。
- **类别不平衡感知** —— 用 `balanced_accuracy` / `macro_f1` 选最优权重；输出每类
  precision/recall/F1、混淆矩阵、预测 CSV 和错误样本清单。
- **专业训练循环** —— AMP、断点续训、早停、warmup + cosine 学习率、标签平滑、
  MixUp/CutMix、EMA、梯度裁剪、完整随机种子、逐 epoch CSV 指标。
- **模型库** —— 注册表背后的 17 个维护版 timm/torchvision 主干 + 38 个 `legacy_*` 键。
- **可解释性与部署** —— Grad-CAM 热力图（`garbage explain`）、TTA、ONNX 导出、
  Gradio 演示。
- **质量门禁** —— ruff lint/format、行为级 pytest 覆盖、GitHub Actions CI
  （Python 3.10–3.12）、pre-commit 钩子。

## 快速开始

```bash
# 1. 创建并激活环境（Python >= 3.10）
uv venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# 2. 下载、校验、应用数据审计补丁，再生成可移植清单
python scripts/download_data.py            # -> data/raw/
garbage prepare-data --set data.data_dir data/raw

# 3. 用稳定运行名训练轻量 MobileNet
garbage train --config configs/resnet50.yaml \
  --set model.name mobilenetv3_small_100 \
  --set train.epochs 3 --set train.batch_size 16 \
  --set data.num_workers 0 --set run_name quickstart-mobilenet

# 4. 在留出的测试集上评测最优 checkpoint
garbage evaluate --checkpoint artifacts/quickstart-mobilenet/best.pt --plot

# 5. 预测单张图 / 整个文件夹
garbage predict --checkpoint artifacts/quickstart-mobilenet/best.pt \
  --image data/raw/paper/paper1.jpg --top-k 3
garbage predict --checkpoint artifacts/quickstart-mobilenet/best.pt --image <文件夹>

# 6. 进阶工具：Grad-CAM 热力图、模型尺寸表
garbage explain --checkpoint artifacts/quickstart-mobilenet/best.pt \
  --image data/raw/paper/paper1.jpg
garbage bench
```

ONNX 与网页演示使用可选依赖：

```bash
uv pip install -e ".[onnx]"
garbage export-onnx --checkpoint artifacts/quickstart-mobilenet/best.pt --output model.onnx

uv pip install -e ".[demo]"
garbage demo --checkpoint artifacts/quickstart-mobilenet/best.pt
```

一行 CPU 冒烟运行（长训练前的快速检查）：

```bash
garbage train --config configs/resnet50.yaml --set model.name mobilenetv3_small_100 \
  --set train.epochs 1 --set train.batch_size 8 --set data.num_workers 0 \
  --set device cpu --set run_name quickstart-smoke
# 或只跑 1 个 batch 验证整条管线：
garbage train --config configs/resnet50.yaml --dry-run
```

## 学习路线（初学者必读）

**不知道从哪开始？跟着 [学习路线](docs/learning-path.zh-CN.md) 走。**
它是 7 个阶段的课程，带你从"什么是图像分类？"到"我亲手训练并评测了自己的模型，
并且能解释报告上每一个数字"—— 每阶段都列出该读什么、该跑什么、练习什么，
并给出预期观察。

| 阶段 | 主题 | 核心收获 |
|---|---|---|
| 1 | 问题与数据 | 类别不平衡、训练/验证/测试集纪律 |
| 2 | 数据管道 | 可移植 manifest、数据增强、为什么按种子切分 |
| 3 | 第一个模型 | timm 主干、模型注册表 |
| 4 | 训练循环 | 损失、优化器、学习率调度、warmup、MixUp、EMA |
| 5 | 评测 | accuracy vs balanced accuracy、F1、混淆矩阵 |
| 6 | 推理与解释 | predict、Grad-CAM、TTA、ONNX |
| 7 | 实验与可复现 | config.yaml 是唯一真相 |

配套文档：[`docs/how-it-works.md`](docs/how-it-works.md) 讲解数据流与训练循环里
每一项技术的原理；[`docs/experiments.zh-CN.md`](docs/experiments.zh-CN.md) 保留旧实验
日志与重跑要求（含"踩坑教训"章节）。

## 配置系统

每个实验是一个 YAML 文件，含三个区块（`data`、`model`、`train`）；任何子集都会
覆盖默认值（见 `src/garbage_classifier/config.py`）。CLI 可用点号键覆盖任意值：

```bash
garbage train --config configs/resnet50.yaml \
  --set train.lr 1e-4 \
  --set model.name swin_tiny_patch4_window7_224 \
  --set train.epochs 80
```

解析后的配置会保存到 `artifacts/<run>/config.yaml`。checkpoint 足以独立推理；
完整复现实验还必须保存当时的 manifest、源数据与审计补丁版本、依赖锁文件，不能只靠
artifact 目录。

## 模型库（timm 主干网络）

| 注册表键 | 参数量* | 说明 |
|---|---:|---|
| `resnet18/34/50/101/152` | 11.7M–60.2M | 经典基线 |
| `convnext_tiny/small` | 28.6M–50.2M | 精度/算力均衡 |
| `mobilenetv3_small_100/large_100` | 2.5M–5.5M | 移动端友好 |
| `efficientnet_b0/b3`, `efficientnetv2_s` | 5.3M–21.5M | FLOPs 优化 |
| `swin_tiny_patch4_window7_224` | 28.3M | Transformer |
| `vit_base_patch16_224` | 86.6M | Transformer |
| `regnetx_002/004` | 1.8M–5.2M | 轻量 |
| `tv_resnet50` | 23.5M | torchvision 对照基线 |

\* timm 的 ImageNet 预训练参数量。完整表格（含 FLOPs）运行 `garbage bench`；
`available_models()` 列出全部注册键。

## 数据集

- 上游 v1 压缩包含 2527 张图。下载器先校验压缩包，再应用 `v1.0-audit.1` 审计补丁，
  删除 3 个已复核的错误标签/重复项，然后按种子 666 分层切分 80/10/10。
- **不平衡**：`trash` 仅占训练集 5.4%，而 `paper` 占 23.5% —— 只看 accuracy 会
  被误导，所以评测总是输出 balanced accuracy、macro/weighted F1 与每类指标。
- 托管在 GitHub Release（不在 git 里）：`garbage-classification.tar.gz`
  （39.7 MB，SHA-256 `be0b99fc…ea1b6`）。用 `python scripts/download_data.py`
  获取（自动校验）。
- 原始来源是 Kaggle 公开数据集 "Garbage classification"。

## 项目结构

```
Image-Classification/
├── src/garbage_classifier/
│   ├── cli.py            CLI 表现层（参数解析 + 分发）
│   ├── config.py         类型化配置（YAML + 点号覆盖）
│   ├── data/             manifest、数据集、变换、prepare-data 逻辑
│   ├── models/           注册表 + timm 模型库（+ legacy 兼容层）
│   │   └── legacy_models/   移植的手写模型（懒加载）
│   ├── training/         Trainer、checkpoint、train 命令逻辑
│   ├── evaluation/       指标、报告、evaluate 命令逻辑
│   ├── inference/        Predictor、Grad-CAM、ONNX 导出、demo
│   └── utils/            种子、设备、git revision、日志
├── configs/              示例实验 YAML
├── scripts/              download_data.py、plot_metrics.py
├── tests/                单元 + CPU 端到端冒烟测试
├── docs/                 学习路线、原理讲解、实验日志
├── data/                 （gitignore）下载的数据集 + 生成的清单
└── artifacts/            （gitignore）每次运行：config.yaml、metrics.csv、best.pt、图
```

## 类别不平衡消融实验

`trash` 仅占训练集 5.4%，因此提供了三种可对比策略（这些配置用 `balanced_accuracy`
选择最优 checkpoint）：

| 配置 | 策略 |
|---|---|
| `configs/imbalance_none.yaml` | 普通 CE loss、均匀采样（基线） |
| `configs/imbalance_weighted_loss.yaml` | loss 中按类别频率反比加权 |
| `configs/imbalance_weighted_sampler.yaml` | `WeightedRandomSampler`（过采样稀有类） |

旧实验数字需要在审计后数据上重跑：见
[`docs/experiments.zh-CN.md`](docs/experiments.zh-CN.md#实验-3类别不平衡三策略对比mobilenetv3-small-预训练15-epochs无-mixup)。

## 可复现性保证

- [x] 全新环境：`pip install -e .` + 一条 CLI 命令 → CPU 冒烟运行
- [x] checkpoint 携带独立推理所需的信息
- [ ] 完整复现实验需归档 manifest、源数据/补丁版本、配置、权重和依赖锁文件
- [x] 推理永远不需要手工重复类别名 / 归一化参数 / 模型名
- [x] CI 对 lint + 单元测试 + 最小训练流程全绿

## 文档索引

| 文档 | 语言 | 用途 |
|---|---|---|
| [`docs/learning-path.md`](docs/learning-path.md) | EN | 初学者 7 阶段课程 |
| [`docs/learning-path.zh-CN.md`](docs/learning-path.zh-CN.md) | 中文 | **初学者 0→1 学习路线（从这里开始）** |
| [`docs/tutorial/`](docs/tutorial/README.zh-CN.md) | 中文 | **实操教程：设备/数据/模型/训练策略** |
| [`docs/how-it-works.zh-CN.md`](docs/how-it-works.zh-CN.md) | 中文 | 数据流 + 每项技术为什么存在 |
| [`docs/experiments.zh-CN.md`](docs/experiments.zh-CN.md) | 中文 | 可复现实验日志 + 教训 |

## 许可证

[MIT](LICENSE)
