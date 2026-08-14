# PyTorch 图像分类实践

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/pytorch-image-classification-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**English version: [README.md](README.md)**

通过一个可以真实运行的 PyTorch 项目，学习从数据准备、训练到评测和错误分析的
完整图像分类流程。示例任务包含 cardboard、glass、metal、paper、plastic 和 trash
六类垃圾图片。

## 这个项目适合谁

主要面向具备 Python 基础，理解 Tensor、loss 和梯度的大致含义，但还没有独立完成过
完整图像分类项目的学习者。不要求会手写 CNN，也不要求读过论文。

推荐路径大约需要 **8–12 小时**。如果 logits、loss 或梯度仍然陌生，可以先读
[深度学习最小概念](docs/tutorial/00-basics.zh-CN.md)。

## 从这里开始

第一次先完成一个短而完整的流程：

```text
准备数据 -> dry run -> 训练 -> 评测 -> 推理
```

创建环境并安装项目：

```bash
uv venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

先确认命令和最小配置可以正常解析：

```bash
garbage --help
garbage show-config --config configs/learning_minimal.yaml
```

下载经过质量检查的公开数据集，并生成固定的训练集、验证集和测试集 manifest：

```bash
python scripts/download_data.py
garbage prepare-data --set data.data_dir data/raw
```

正式训练前只运行一个 batch：

```bash
garbage train --config configs/learning_minimal.yaml --dry-run
```

日志中应该出现 `dry-run OK`，同时显示输入形状、输出形状和 loss。这说明数据加载、
模型、前向计算、损失函数和反向传播能够连起来运行，但还不代表模型已经学会分类。

接着训练两个 epoch，得到真实的 checkpoint 和指标记录：

```bash
garbage train --config configs/learning_minimal.yaml \
  --set train.epochs 2 --set run_name first-run
```

`artifacts/first-run/` 下的文件回答不同问题：

- `config.yaml`：最终生效的配置是什么？
- `metrics.csv`：训练和验证指标怎样随 epoch 变化？
- `best.pt`：评测和推理应该使用哪个 checkpoint？

## 看看训练结果

在留出的测试集上评测，再预测一张图片：

```bash
garbage evaluate --checkpoint artifacts/first-run/best.pt --plot
garbage predict --checkpoint artifacts/first-run/best.pt \
  --image data/raw/paper/paper1.jpg --top-k 3
```

先别只看 accuracy。比较 balanced accuracy 和 macro F1，看看哪些类别的指标较弱，
再打开 `errors.csv` 检查错误样本。两个 epoch 的结果可能并不好，它的作用是让完整流程
先运转起来，同时保持每一步都容易观察。

推理会从 checkpoint 恢复模型名称、类别和预处理参数，不需要再次手工填写训练配置。

![ResNet50 参考实验的训练曲线](docs/assets/reference-resnet50-curves.png)

一次经过核对的 15 epoch 运行在测试集上得到 93.33% accuracy。可以继续查看
[参考运行说明](docs/reference-run.zh-CN.md)，其中保留了准确配置、运行环境、各类别结果和
这个数字的适用范围。

## 接着读什么

- 按[学习路线](docs/learning-path.zh-CN.md)继续，逐步理解每个环节，并做几组短小的
  对比。
- 用[代码导览](docs/code-tour.zh-CN.md)追踪数据、配置、训练、评测和推理，不必从头
  顺序阅读所有文件。
- 某个设计选择需要进一步解释时，查看[原理说明](docs/how-it-works.zh-CN.md)。
- 想比较训练策略时，查看[实验说明](docs/experiments.zh-CN.md)，每次只改变一个因素。
- 需要更细的操作说明时，查看[实操教程](docs/tutorial/README.zh-CN.md)。

## 换成自己的数据

只要每个类别使用一个文件夹，同一套流程就可以处理其他图片数据。
[自己的数据指南](docs/using-your-data.zh-CN.md)说明了怎样预览图片、使用独立 manifest、
运行短基线并查看错误，同时避免覆盖示例数据的固定切分。

## 项目覆盖的内容

核心路径包括：

- 经过检查、可以复现的训练集、验证集和测试集 manifest；
- 默认值清晰、支持命令行覆盖的 YAML 配置；
- 包含 checkpoint 和断点续训的可读训练循环；
- 每类 precision、recall、F1、balanced accuracy、混淆矩阵和错误样本；
- 使用自包含 checkpoint 的单图与批量预测。

理解核心流程后，还可以按需使用预训练 CNN 与 Transformer、AMP、早停、
MixUp/CutMix、EMA、类别不平衡策略、TTA、Grad-CAM、ONNX 和 Gradio。

```bash
garbage explain --checkpoint artifacts/first-run/best.pt \
  --image data/raw/paper/paper1.jpg
garbage bench

uv pip install -e ".[onnx]"
garbage export-onnx --checkpoint artifacts/first-run/best.pt --output model.onnx

uv pip install -e ".[demo]"
garbage demo --checkpoint artifacts/first-run/best.pt
```

## 配置与模型

配置按下面的顺序生效：

```text
代码默认值 < YAML 文件 < --set 覆盖 < 专用命令行参数
```

[配置参数参考](docs/config-reference.zh-CN.md)解释了所有字段。最小配置关闭高级训练技巧，
让第一次训练时更容易看清基础过程。模型注册表还提供由 timm 和 torchvision 维护的
ResNet、MobileNetV3、EfficientNet、ConvNeXt、RegNet、Swin 和 ViT。运行
`garbage bench` 可以比较参数量，以及在依赖可用时查看 FLOPs。

## 项目结构

```text
examples/                    适合先运行和阅读的 5 个小程序
configs/                     可直接运行的实验配置
docs/                        学习路线、教程和代码导览
scripts/                     下载、预览和绘图工具
src/garbage_classifier/      安装后运行的应用和可复用代码包
tests/                       行为测试和进阶参考
data/                        本地数据集与 manifest（gitignore）
artifacts/                   本地运行产物与 checkpoint（gitignore）
```

学习时可以从 `examples/` 开始，再跟着文档进入 `src/`。准备贡献代码时，先读
[贡献指南](CONTRIBUTING.zh-CN.md)，以及 [scripts](scripts/README.zh-CN.md)、
[src](src/README.zh-CN.md) 和 [tests](tests/README.zh-CN.md) 的目录说明。

## 文档索引

- [学习路线](docs/learning-path.zh-CN.md) · [English learning path](docs/learning-path.md)
- [代码导览](docs/code-tour.zh-CN.md) · [English code tour](docs/code-tour.md)
- [配置参数参考](docs/config-reference.zh-CN.md) · [English reference](docs/config-reference.md)
- [原理说明](docs/how-it-works.zh-CN.md) · [English guide](docs/how-it-works.md)
- [实验说明](docs/experiments.zh-CN.md) · [English experiments](docs/experiments.md)
- [参考运行](docs/reference-run.zh-CN.md) · [English reference run](docs/reference-run.md)
- [实操教程](docs/tutorial/README.zh-CN.md)

## 许可证

[MIT](LICENSE)
