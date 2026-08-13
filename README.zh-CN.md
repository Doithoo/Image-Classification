# 垃圾图像分类（Garbage Image Classification）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/Image-Classification/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**English version: [README.md](README.md)**

这是一个从数据准备、训练、评测到部署的 PyTorch 图像分类学习项目。示例任务包含
六类垃圾：cardboard、glass、metal、paper、plastic 和 trash。

本项目按“边运行边学习”的方式设计。第一次使用时，不需要先看懂所有文件，也不需要
先掌握所有训练技巧。

## 从这里开始

第一次使用建议严格按照下面的顺序：

    README -> 学习路线 -> 最小配置 -> 训练 -> 评测 -> 推理

1. 先读[学习路线](docs/learning-path.zh-CN.md)，它会告诉你每个阶段该学什么、运行
   什么、观察什么。
2. 安装项目，使用[最小配置](configs/learning_minimal.yaml)完成第一次训练。
3. 先训练 2～3 个 epoch，读懂评测报告，再预测一张图片。
4. 完成第一次闭环后，再读[代码导览](docs/code-tour.zh-CN.md)，每次只修改一个配置项。

项目默认你会一些 Python。如果还不熟悉 Tensor、logits、loss 或梯度，先看
[深度学习最小概念](docs/tutorial/00-basics.zh-CN.md)。

## 项目包含什么

- 可复现的训练集、验证集和测试集 manifest，以及数据质量检查。
- 默认值清晰可见、支持命令行覆盖的 YAML 实验配置。
- 维护中的 CNN 和 Transformer 主干网络，以及可选的预训练权重。
- 包含 checkpoint、断点续训、AMP、早停、MixUp/CutMix、EMA 和类别不平衡策略的
  可读训练循环。
- 不只看 accuracy，同时输出每类 precision、recall、F1、balanced accuracy、
  混淆矩阵和错误样本。
- 单图与批量预测、TTA、Grad-CAM、ONNX 导出和 Gradio 演示。

## 安装环境

支持 Python 3.10、3.11 和 3.12。下面使用 uv，也可以使用普通虚拟环境和
pip install -e ".[dev]"。

    uv venv
    source .venv/bin/activate              # Windows: .venv\Scripts\activate
    uv pip install -e ".[dev]"

先确认命令可用，再下载数据：

    garbage --help
    garbage show-config --config configs/learning_minimal.yaml

## 第一次实验

下载公开数据集、生成可移植 manifest，然后运行一次短训练。下载脚本会先校验压缩包，
再应用数据质量修正；下载的数据和训练产物只保留在本地，不会提交进仓库。

    python scripts/download_data.py
    garbage prepare-data --set data.data_dir data/raw
    garbage train --config configs/learning_minimal.yaml --dry-run

    garbage train --config configs/learning_minimal.yaml \
      --set train.epochs 2 --set run_name first-run

运行产物会写入 artifacts/first-run/，其中包括配置、指标和 checkpoint。开始长训练
前，可以先查看最终生效的配置：

    garbage show-config --config configs/learning_minimal.yaml \
      --set train.epochs 5 --device cpu

## 使用 checkpoint

评测会使用训练时留出的测试集；推理会从 checkpoint 恢复类别名和预处理参数，不需要
再次手工填写训练配置。

    garbage evaluate --checkpoint artifacts/first-run/best.pt --plot
    garbage predict --checkpoint artifacts/first-run/best.pt \
      --image data/raw/paper/paper1.jpg --top-k 3

完成基础闭环后，再尝试这些可选工具：

    garbage explain --checkpoint artifacts/first-run/best.pt \
      --image data/raw/paper/paper1.jpg
    garbage bench

    uv pip install -e ".[onnx]"
    garbage export-onnx --checkpoint artifacts/first-run/best.pt --output model.onnx

    uv pip install -e ".[demo]"
    garbage demo --checkpoint artifacts/first-run/best.pt

## 你将学会什么

| 阶段 | 要回答的问题 | 主要产出 |
|---|---|---|
| 1. 数据 | 分类什么，应该怎样切分？ | 经过检查的 manifest 和类别统计 |
| 2. 数据管道 | 图片如何变成 Tensor batch？ | Dataset、变换和 DataLoader |
| 3. 模型 | backbone、分类头和注册表是什么？ | 可配置的预训练分类器 |
| 4. 训练 | loss、梯度和学习率怎样改变权重？ | 可复现的训练循环 |
| 5. 评测 | 为什么不能只看 accuracy？ | 每类指标和混淆矩阵 |
| 6. 推理 | 训练好的模型怎样真正使用？ | 单图、批量和 TTA 预测 |
| 7. 部署 | 其他程序怎样调用模型？ | Grad-CAM、ONNX 和 Gradio |

完整课程见[学习路线](docs/learning-path.zh-CN.md)；需要某一步的详细操作时，查阅
[实操教程](docs/tutorial/README.zh-CN.md)。

## 配置与模型

实验参数写在 YAML 中，生效顺序如下：

    代码默认值 < YAML 文件 < --set 覆盖 < 专用命令行参数

所有字段的含义、默认值和相互关系见[配置参数参考](docs/config-reference.zh-CN.md)。
[最小配置](configs/learning_minimal.yaml)关闭了大多数进阶技巧，适合第一次读训练循环。

模型注册表提供维护中的 timm 和 torchvision 主干网络：

- resnet18、resnet34、resnet50、resnet101、resnet152
- mobilenetv3_small_100、mobilenetv3_large_100
- efficientnet_b0、efficientnet_b3、efficientnetv2_s
- convnext_tiny、convnext_small、regnetx_002、regnetx_004
- swin_tiny_patch4_window7_224、vit_base_patch16_224、tv_resnet50

运行 garbage bench 可以比较参数量，以及在依赖可用时显示 FLOPs。

## 项目结构

    examples/                    适合先运行和阅读的 5 个小程序
    configs/                     可直接运行的实验配置
    docs/                        学习路线、教程和代码导览
    scripts/                     下载、预览和绘图工具
    src/garbage_classifier/      安装后运行的应用和可复用代码包
    tests/                       行为测试和进阶参考
    data/                        本地数据集与 manifest（gitignore）
    artifacts/                   本地运行产物与 checkpoint（gitignore）

学习时按 examples/ -> docs/ -> src/ 阅读；贡献代码时阅读
[贡献指南](CONTRIBUTING.zh-CN.md)，再查看 tests/、pyproject.toml 和 CI。三个代码目录
各有说明：[scripts](scripts/README.zh-CN.md)、[src](src/README.zh-CN.md)、
[tests](tests/README.zh-CN.md)。

## 文档索引

- [学习路线](docs/learning-path.zh-CN.md) · [English learning path](docs/learning-path.md)
- [代码导览](docs/code-tour.zh-CN.md) · [English code tour](docs/code-tour.md)
- [配置参数参考](docs/config-reference.zh-CN.md) · [English reference](docs/config-reference.md)
- [原理说明](docs/how-it-works.zh-CN.md) · [English guide](docs/how-it-works.md)
- [实验记录](docs/experiments.zh-CN.md) · [English experiments](docs/experiments.md)
- [实操教程](docs/tutorial/README.zh-CN.md)

## 许可证

[MIT](LICENSE)
