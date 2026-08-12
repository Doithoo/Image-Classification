# 图像分类学习路线（0 → 1）

> 本路线面向初学者，用这个项目作为脚手架，从零学会"如何训练图像分类模型"。
> 全程约 **8–12 小时**（含动手实验），按顺序完成即可。
>
> **English version: [learning-path.md](learning-path.md)** ·
> **卡住了？配套实操教程：[教程系列](tutorial/README.zh-CN.md)**
> （不同设备怎么跑、数据怎么备、模型怎么建、超参怎么选）

## 前置知识（可跳过，但建议扫一眼）

- Python 基础：函数、类、循环、`import`
- PyTorch 基础：张量（tensor）、`nn.Module`、`optimizer.step()` 的直觉理解
  （推荐先花 1 小时跑一遍 [PyTorch 60 分钟入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)）
- 一点机器学习概念：训练集/验证集/测试集、过拟合、准确率

**本路线不要求**：读过论文、懂数学推导、会写 CNN。

---

## 阶段 0：环境准备（约 15 分钟）

**目标**：把项目跑起来，看到第一条训练日志。

```bash
cd Image-Classification
uv pip install -e ".[dev]"            # 安装依赖
python scripts/download_data.py       # 下载数据集（自动校验 SHA-256）
garbage prepare-data --set data.data_dir data/raw   # 生成清单
garbage train --config configs/resnet50.yaml --dry-run  # 1 个 batch 冒烟
```

**要观察的**：
- `data/manifests/` 下生成了 `train.csv / valid.csv / test.csv`（含 `summary.txt`）
- `--dry-run` 输出 `dry-run OK: input=(32,3,224,224) output=(32,6) loss=1.6xx`
  —— 这代表：数据能读、模型能建、前向反向都能跑。

**读什么**：`README.zh-CN.md` 的「快速开始」和「项目结构」。

**验证题**：`data/manifests/summary.txt` 里三个切分各多少张？每类几张？
（答案应与文档一致：train 2019 / valid 252 / test 256）

---

## 阶段 1：问题与数据（约 45 分钟）

**目标**：理解任务本身 —— 6 类垃圾图片，以及为什么"只看准确率"会骗人。

**读什么**：
- [`docs/how-it-works.zh-CN.md`](how-it-works.zh-CN.md) 第 1–2 节（数据流总览、为什么用 CSV manifest）
- `src/garbage_classifier/data/manifest.py` 顶部注释
- 打开 `data/manifests/summary.txt` 看各类数量

**动手**：
```bash
python scripts/plot_metrics.py  # 先不用，阶段 4 再说
ls data/raw/                    # 看每个类别文件夹
```

**核心知识点（务必理解）**：
1. **类别不平衡**：`trash` 只有 109 张训练图（5.4%），`paper` 有 475 张（23.5%）。
   一个"永远猜 paper"的模型准确率也有 23.5% —— 所以后面我们用
   `balanced_accuracy` / `macro_f1`，而不是只看 accuracy。
2. **训练/验证/测试三分**：验证集用于选模型，测试集只用于最终评估，**绝不碰**。
   这是防止"自欺欺人"的纪律。
3. **为什么按种子切分**：`seed=666` 保证任何人跑 `prepare-data` 都得到同一份
   切分 —— 这是"可复现实验"的第一步。

**验证题**：如果模型把 test 集 256 张全猜成 `paper`，accuracy 是多少？
balanced_accuracy 呢？（答案：0.234 和 0.167 —— 后者揭示了模型的"无用"）

---

## 阶段 2：数据管道（约 1 小时）

**目标**：弄清"图片 → 张量 → 批次"的完整旅程。

**读什么**：
- `src/garbage_classifier/data/dataset.py`（Dataset 如何工作）
- `src/garbage_classifier/data/transforms.py`（每行变换的用途）
- `src/garbage_classifier/data/manifest.py` 的 `build_manifest` / `load_manifest`

**动手**：
```bash
garbage prepare-data --set data.data_dir data/raw --strict   # 加重复检测
python -c "
from garbage_classifier.data import ImageClassificationDataset
from garbage_classifier.data.transforms import build_train_transform
from garbage_classifier.config import load_config
ds = ImageClassificationDataset('data/manifests/train.csv',
      transform=build_train_transform(load_config().data))
img, label = ds[0]
print(img.shape, label)   # torch.Size([3, 224, 224]) 0
"
```

**核心知识点**：
1. **为什么是 CSV manifest 而不是直接遍历文件夹**：相对路径 + 固定切分 =
   可移植 + 可复现。旧版项目用 Windows 反斜杠路径，换系统就崩 —— 这是重构的
   第一个动机。
2. **为什么训练集和验证集用不同的变换**：训练要多样性（随机裁剪/翻转/增强），
   验证要稳定（固定居中裁剪）。`transforms.py` 里 `RandomResizedCrop` vs
   `CenterCrop` 就是这对关系的体现。
3. **归一化**：`Normalize(mean, std)` 把像素从 [0,1] 变成近似标准正态 —— 帮助
   模型收敛。mean/std 由 `03_compute_stats`（旧版脚本）或直接计算得到，训练/推理
   必须用同一份（我们的 checkpoint 自带了它，见阶段 6）。

**验证题**：为什么验证集 DataLoader 的 `shuffle=False`？（提示：评测结果要确定）

---

## 阶段 3：第一个模型（约 45 分钟）

**目标**：理解"模型注册表"和主干网络（backbone）的概念。

**读什么**：
- `src/garbage_classifier/models/registry.py`（注册表怎么工作）
- `src/garbage_classifier/models/zoo.py`（timm 如何接入）
- `src/garbage_classifier/config.py` 的 `ModelConfig`

**动手**：
```bash
garbage bench                          # 看每个模型多大（参数量 + FLOPs）
garbage train --config configs/resnet50.yaml --set train.epochs 1 --set device cpu
```

**核心知识点**：
1. **backbone（主干）**：ResNet50 等预训练网络负责提取特征，最后接一个新的
   分类头（把 1000 类 ImageNet 输出换成我们的 6 类）。`num_classes=6` 就是干这个。
2. **预训练 vs 从零**：`pretrained: true` 用 ImageNet 上训好的权重初始化 —— 小
   数据集上通常大幅优于从零（`docs/experiments.zh-CN.md` 实验 2 有实测：+27 个点）。
3. **为什么用注册表**：`config.yaml` 里写 `model.name` 就能换模型，代码不用动。
   这叫"配置驱动" —— 对比旧版（改代码切换模型）是本质区别。

**验证题**：`garbage bench` 里 mobilenetv3_small 的参数量约为 resnet50 的几分之几？
猜一下两者精度差多少？（实测见实验 2：约差 12 个点）

---

## 阶段 4：训练循环（约 2 小时，核心阶段）

**目标**：真正理解一次训练迭代里发生了什么。

**读什么（配合代码逐行看）**：
- `src/garbage_classifier/training/trainer.py` —— 重点看 `_run_epoch` 和 `fit`
- `src/garbage_classifier/training/mixup.py` 顶部注释（MixUp/CutMix 原理）
- `src/garbage_classifier/training/ema.py` 顶部注释（EMA 原理）
- `src/garbage_classifier/training/weights.py`（类别不平衡的两种处理）

**动手（三组对比实验，每组约 6 分钟）**：
```bash
# A. 基线
garbage train --config configs/imbalance_none.yaml --set model.name mobilenetv3_small_100 \
  --set train.epochs 15 --set train.mixup_alpha 0 --set data.num_workers 0 --set run_name learn-a

# B. 打开 MixUp
garbage train --config configs/imbalance_none.yaml --set model.name mobilenetv3_small_100 \
  --set train.epochs 15 --set train.mixup_alpha 0.2 --set data.num_workers 0 --set run_name learn-b

# 对比两条训练曲线
python scripts/plot_metrics.py artifacts/learn-a/metrics.csv
python scripts/plot_metrics.py artifacts/learn-b/metrics.csv
```

**核心知识点（训练循环的心智模型）**：
```
每个 epoch：
  对每个 batch：
    outputs = model(images)     # 前向：预测
    loss = loss_fn(outputs, labels)  # 比较预测与真相
    loss.backward()             # 反向：算梯度
    optimizer.step()            # 沿梯度更新权重
  scheduler.step()              # 学习率退火
  在验证集上评测 → 记录指标 → 更新 best.pt
  若 N 个 epoch 没进步 → early stop
```

1. **warmup + cosine**：学习率从 1% 线性爬升到目标值，再按余弦退火到接近 0。
   为什么？开局权重离最优远，大学习率会"冲过头"；后期小学习率让权重稳定落位。
2. **MixUp**：把两张图按 λ 混合（标签也混合）—— 模型被迫学特征而非死记，
   典型的正则化。**注意**：MixUp 后训练 loss 会偏高，但验证集反而更好 ——
   别被训练 loss 骗了。
3. **EMA**：维护一个"影子权重"（历史权重的指数平均），验证和保存用影子、
   训练用快权重。影子更平滑、泛化更好。**但是**：decay 的时间常数是
   1/(1−decay) 步，短训练里 `0.999` 根本追不上 —— 这就是实验 4 的教训。
4. **checkpoint 自包含**：best.pt 里存的不只是权重，还有配置、类别表、优化器
   状态 —— 所以断点续训（`--resume`）和复现实验才可能。

**验证题**：为什么 MixUp 下训练 loss 偏高是"正常且健康"的？打开两条曲线的
train_loss 和 val_loss 对比说明。

---

## 阶段 5：评测（约 1 小时）

**目标**：读懂一张评测报告，能解释每个数字。

**读什么**：
- `src/garbage_classifier/evaluation/metrics.py`（每个指标怎么算）
- `docs/experiments.zh-CN.md` 实验 1（完整报告示例）

**动手**：
```bash
garbage evaluate --checkpoint artifacts/learn-b/best.pt --plot
# 生成 classification report + confusion_matrix.png + predictions.csv + errors.csv
```

**核心知识点**：
1. **accuracy 的局限**：256 张里 60 张 paper，全猜 paper 也有 23.4%。
2. **balanced_accuracy**：各类 recall 的平均 —— 每类平等对待。
3. **precision / recall / F1**（以 trash 为例）：
   - precision = 预测为 trash 里真的有多少是 trash（"猜的准不准"）
   - recall = 真的 trash 里被找出来多少（"漏没漏"）
   - F1 = 两者的调和平均
   - 重平衡策略就是在 precision↔recall 之间做权衡（实验 3 的教训！）
4. **混淆矩阵**：行=真实、列=预测。对角线越亮越好；看"垃圾"常被错成哪类，
   就是错误分析的第一步。

**验证题**：errors.csv 里挑 3 个错误样本，用阶段 6 的 Grad-CAM 看模型为什么错。

---

## 阶段 6：推理与解释（约 45 分钟）

**目标**：理解"推理只看 checkpoint"和 Grad-CAM。

**读什么**：
- `src/garbage_classifier/inference/predictor.py`（Predictor 如何从 checkpoint 恢复一切）
- `src/garbage_classifier/inference/gradcam.py` 顶部注释（Grad-CAM 原理）

**动手**：
```bash
garbage predict --checkpoint artifacts/learn-b/best.pt --image data/raw/paper/paper1.jpg --top-k 3
garbage predict --checkpoint artifacts/learn-b/best.pt --image data/raw/paper/paper1.jpg --top-k 3 --tta
garbage explain --checkpoint artifacts/learn-b/best.pt --image data/raw/paper/paper1.jpg --output gradcam.png
```

**核心知识点**：
1. **checkpoint 自包含的威力**：`predict` 不需要你告诉它类别名、均值方差、模型名
   —— 全部从 best.pt 恢复。训练/推理配置永不漂移。
2. **TTA（测试时增强）**：推理时把图水平翻转再推一次，两次 softmax 取平均。
   不训练任何东西，白捡 1–2 个点（实验 5：0.906 → 0.922）。
3. **Grad-CAM**：用最后一层卷积的特征图 × 类别得分的梯度，加权得到"模型看哪里"
   的热力图。模型分错时，看它看的是不是错误区域 —— 错误分析第一利器。

**验证题**：`--tta` 前后 top-1 概率有什么变化？解释为什么平均会改变置信度。

---

## 阶段 7：实验与可复现（约 1 小时）

**目标**：把"做实验"变成纪律，理解可复现实验平台的本质。

**读什么**：
- `docs/experiments.zh-CN.md` 全文（尤其是每节的"学习要点"）
- `src/garbage_classifier/training/checkpoint.py`（checkpoint 里存了什么）

**动手（完成一个自己的小实验矩阵）**：
```bash
# 你的第一个消融：lr 对比
garbage train --config configs/resnet50.yaml --set train.lr 1e-3 --set train.epochs 10 \
  --set data.num_workers 0 --set run_name myexp-lr1e3
garbage train --config configs/resnet50.yaml --set train.lr 1e-4 --set train.epochs 10 \
  --set data.num_workers 0 --set run_name myexp-lr1e4
python scripts/plot_metrics.py artifacts/myexp-lr1e3/metrics.csv
python scripts/plot_metrics.py artifacts/myexp-lr1e4/metrics.csv
```

**核心知识点（学习项目的"职业习惯"）**：
1. **每次实验 = 一个 artifact 目录**：`config.yaml`（完整配置）+ `metrics.csv`
   （逐 epoch 指标）+ `best.pt`（可复现权重）。**config.yaml 是唯一真相** ——
   本项目早期就把"预训练微调"误记成"从零训练"，全靠 config.yaml 纠正。
2. **一次只改一个变量**：对比实验必须保证"只有自变量不同"，否则结论无效。
3. **先记录，再下结论**：直觉（"重平衡应该更好"）经常被数据打脸（实验 3）。

**验证题**：写完 `docs/experiments.zh-CN.md` 风格的实验记录（可以就用 myexp 两个 run），
包含：命令、表格、一个"学习要点"。

---

## 毕业项目（可选，约 2–3 小时）

用学到的全部技能完成一个完整闭环：

1. 换一个新模型（`convnext_tiny`）跑一个完整训练（20+ epoch）
2. 对比它与 resnet50 的精度/速度
3. 用 Grad-CAM 分析它最容易错的两个类别
4. 导出 ONNX 并用 `garbage demo` 做一个可交互的演示
5. 把过程和结论写成 `docs/my-experiment.md`

---

## 学习地图（所有文档和代码的索引）

```
开始 → README.zh-CN.md（快速开始）
  ↓
阶段 0-2 → docs/how-it-works.zh-CN.md（数据流）＋ data/ 目录代码
  ↓
阶段 3-4 → models/registry.py ＋ training/trainer.py ＋ training/mixup.py ＋ training/ema.py
  ↓
阶段 5-6 → evaluation/metrics.py ＋ inference/predictor.py ＋ inference/gradcam.py
  ↓
阶段 7 → docs/experiments.zh-CN.md（前人的实验与教训）＋ training/checkpoint.py
  ↓
毕业项目 → 用 garbage 命令自由组合
```

**记住**：读代码时，先读文件顶部的注释（每个学习模块都写了"为什么"）；
跑实验时，先 `--dry-run`；下结论时，先看 `config.yaml`。
