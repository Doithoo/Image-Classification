# 图像分类学习路线

这条路线面向已经接触过 Python、Tensor、loss 和梯度，但还没有完整做过图像分类项目的
学习者。按顺序阅读和运行大约需要 **8–12 小时**。第一次不必理解所有训练技巧，先让
数据、模型、评测和推理连起来，再回头看细节。

如果 logits、loss 或梯度仍然陌生，先读
[深度学习最小概念](tutorial/00-basics.zh-CN.md)。环境或设备问题可以查阅
[设备与环境教程](tutorial/01-environment.zh-CN.md)。

## 开始之前

确认 Python 版本和项目命令：

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
garbage --version
garbage show-config --config configs/learning_minimal.yaml
```

`show-config` 展示最终生效的完整配置。后面调整参数时，先看这里比凭记忆判断更可靠。

## 先运行一次

下载数据、生成固定切分，再用一个 batch 检查完整管道：

```bash
python scripts/download_data.py
garbage prepare-data --set data.data_dir data/raw
garbage train --config configs/learning_minimal.yaml --dry-run
```

日志中的 `dry-run OK` 表示图片可以加载、模型可以建立、前向与反向计算都能完成。
这一步只检查管道，不评价精度。

接着运行两个 epoch：

```bash
garbage train --config configs/learning_minimal.yaml \
  --set train.epochs 2 --set run_name first-minimal-run
```

先打开 `artifacts/first-minimal-run/config.yaml` 和 `metrics.csv`。配置说明这次实验到底
用了什么，指标文件说明两个 epoch 之间发生了怎样的变化。

## 看看数据

模型开始训练之前，先确认分类对象和标签没有问题：

```bash
python scripts/preview_dataset.py --data-dir data/raw
cat data/manifests/summary.txt
```

预览图适合发现放错目录、内容异常或类别含义不清的图片。`summary.txt` 记录类别、切分
数量、随机种子和 manifest 校验和。

这里值得留意三件事：

- `trash` 样本明显少于其他类别，单看 accuracy 容易忽略它。
- 验证集用于比较配置，测试集只在选定方案后评测。
- 固定种子让同一份数据得到相同切分，实验才有可比性。

进一步的实现细节在 [原理说明](how-it-works.zh-CN.md) 的数据部分和
`src/garbage_classifier/data/manifest.py`。

## 图片怎样变成批次

一张图片进入训练循环前会经过 Dataset、变换和 DataLoader。建议按下面顺序阅读：

```text
src/garbage_classifier/data/manifest.py
src/garbage_classifier/data/dataset.py
src/garbage_classifier/data/transforms.py
```

可以直接查看一个样本：

```bash
python -c "
from garbage_classifier.data import ImageClassificationDataset
from garbage_classifier.data.transforms import build_train_transform
from garbage_classifier.config import load_config
cfg = load_config('configs/learning_minimal.yaml')
ds = ImageClassificationDataset('data/manifests/train.csv',
      transform=build_train_transform(cfg.data))
image, label = ds[0]
print(image.shape, label)
"
```

输出应是 `[3, 224, 224]` 的张量和一个整数标签。训练变换带随机性，验证变换保持
确定性；checkpoint 保存预处理参数，避免训练与推理使用不同的归一化方式。

## 模型怎样产生预测

先读 `src/garbage_classifier/models/registry.py` 和 `models/zoo.py`。配置中的
`model.name` 选择 backbone，manifest 中的类别表决定分类头输出维度。

```bash
garbage bench
garbage show-config --config configs/learning_minimal.yaml
```

`bench` 适合比较参数量，不代表参数更多就一定更适合当前数据。预训练模型通常能在小
数据集上更快得到可用结果，从零训练则更容易看清随机初始化与数据规模的影响。

更完整的模型解释见[模型构建与定制](tutorial/03-model.zh-CN.md)。

## 训练循环在做什么

先运行最小示例，再读正式训练器：

```bash
python examples/03_minimal_training.py
```

```text
每个 batch：
  outputs = model(images)
  loss = loss_fn(outputs, labels)
  loss.backward()
  optimizer.step()

每个 epoch：
  在验证集上计算指标
  更新 best.pt 与 last.pt
  记录 metrics.csv
```

正式实现位于 `src/garbage_classifier/training/trainer.py`。先看 `_run_epoch` 中标记的
基础循环，再看 `fit` 如何加入验证、checkpoint、学习率调度和早停。最小配置关闭
MixUp、EMA 和 AMP，第一次阅读时不会被可选分支打断。

训练 loss 与验证指标回答不同问题。训练 loss 降低说明模型正在适应训练数据；验证
指标停滞或变差则可能意味着过拟合、数据问题或配置不合适。

## 怎样判断模型表现

使用最佳 checkpoint 评测留出的测试集：

```bash
garbage evaluate \
  --checkpoint artifacts/first-minimal-run/best.pt \
  --plot
```

评测会输出 accuracy、balanced accuracy、macro F1、每类 precision/recall/F1、
混淆矩阵、`predictions.csv` 和 `errors.csv`。

先比较 accuracy 与 balanced accuracy。如果两者差距较大，模型可能忽略了样本较少的
类别。再看每类 recall 和混淆矩阵，确认错误集中在哪里。最后打开 `errors.csv`，判断
问题更像标签、图片质量、类别边界还是模型能力。

指标计算在 `src/garbage_classifier/evaluation/metrics.py`，更详细的解释在
[原理说明](how-it-works.zh-CN.md)。

## 怎样使用 checkpoint

checkpoint 不只保存权重，还包含模型名称、类别表、预处理参数和训练配置。因此推理时
只需提供 checkpoint 与图片：

```bash
garbage predict \
  --checkpoint artifacts/first-minimal-run/best.pt \
  --image data/raw/paper/paper1.jpg \
  --top-k 3
```

预测错误时可以生成 Grad-CAM，看看模型关注了哪里：

```bash
garbage explain \
  --checkpoint artifacts/first-minimal-run/best.pt \
  --image data/raw/paper/paper1.jpg \
  --output artifacts/first-minimal-run/gradcam.png
```

Grad-CAM 只能提供线索，不能证明模型的因果理由。热区落在背景或无关物体上时，才值得
进一步检查数据与增强方式。

## 做几组有意义的对比

完整流程跑通后，可以选择一个问题做短对比。每次只改变一个因素，其他配置、manifest
和随机种子保持一致。

例如比较预训练与从零训练：

```bash
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 \
  --set model.pretrained true \
  --set train.epochs 15 \
  --set run_name compare-pretrained

garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 \
  --set model.pretrained false \
  --set train.epochs 15 \
  --set run_name compare-scratch
```

先猜测哪一组收敛更快，再比较最佳验证集 balanced accuracy 和 macro F1。结果只说明
这份数据、这套配置和这个运行环境下发生了什么，不应直接推广到所有任务。

更多可运行的对比见[实验说明](experiments.zh-CN.md)。训练曲线异常时可以查阅
[训练策略选择](tutorial/04-training.zh-CN.md)。

## 接下来可以怎么做

- 想换成另一个按文件夹分类的数据集，查看[自己的数据指南](using-your-data.zh-CN.md)。
- 想看清代码之间怎样连接，打开[代码导览](code-tour.zh-CN.md)。
- 想进一步理解配置覆盖关系，打开[配置流说明](configuration-flow.zh-CN.md)。
- 想尝试 TTA、Grad-CAM、ONNX 或 Gradio，查看
  [推理与部署](tutorial/05-inference-deployment.zh-CN.md)。
- 想了解每个配置字段，查看[配置参数参考](config-reference.zh-CN.md)。

不必一次读完所有内容。遇到具体问题时回到对应部分，通常比顺序通读整个源码更有效。
