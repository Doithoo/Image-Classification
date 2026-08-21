# PyTorch 图像分类实践

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)

**English: [README.md](README.md)**

这是一个面向初学者、强调可复现性的 PyTorch 图像分类项目。示例任务包含纸板、玻璃、金属、纸张、塑料和其他垃圾六类；同一套流程也适用于任意“每类一个文件夹”的图像数据集。

## 从这里开始

安装项目、下载已审计的示例数据后，先跑通最短完整链路：

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py
uv run garbage prepare-data --set data.data_dir data/raw
uv run garbage verify-data --set data.data_dir data/raw
uv run garbage train --config configs/learning_minimal.yaml --dry-run --set data.data_dir data/raw
```

`--dry-run` 会验证数据加载、模型构建、前向、损失和反向传播，但不会创建运行目录或 checkpoint。确认通过后，再创建独立训练运行：

```bash
uv run garbage train --config configs/learning_minimal.yaml \
  --set data.data_dir data/raw --set run_name first-run
uv run garbage evaluate --checkpoint artifacts/first-run/best.pt \
  --set data.data_dir data/raw --plot
uv run garbage predict --checkpoint artifacts/first-run/best.pt \
  --image data/raw/paper/paper1.jpg --top-k 3
```

每个运行都会记录最终配置、数据身份、环境、指标和带版本的安全 checkpoint。评测证据写入 `artifacts/<run>/evaluation/<split>/`，默认不会覆盖已有结果。

![Reference ResNet50 training curves](docs/assets/reference-resnet50-curves.png)

[参考运行记录](docs/recorded-run/README.zh-CN.md) 在固定六类数据划分上得到 93.33% 测试准确率。这是一个有边界的可复现记录，不是通用基准。

## 文档

请从[文档首页](docs/README.zh-CN.md)按目标进入：

- [教程](docs/tutorial/README.zh-CN.md)：环境、数据、模型、训练与推理的第一条路径。
- [概念](docs/concepts/classification-flow.zh-CN.md)：完整数据流、代码导览和配置流。
- [指南](docs/guides/using-your-data.zh-CN.md)：自己的数据、实验、模型选择和排障。
- [参考](docs/reference/config-reference.zh-CN.md)：配置字段、数据身份、指标和 checkpoint 协议。
- [参考运行](docs/recorded-run/README.zh-CN.md)：已发布结果的机器可读证据。

[配置目录](configs/README.zh-CN.md)和[示例目录](examples/README.zh-CN.md)说明每个可运行文件的目的。

## 开发

```bash
uv sync --locked --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv build
uv run twine check dist/*
```

修改前请阅读 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。项目使用 [MIT License](LICENSE)。
