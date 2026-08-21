# 可复现对比指南

下面几组对比都是可选的，用来回答一个具体问题，同时避免一次改变很多设置。checkpoint
由验证集选出，测试集留到确定最终设置后再使用。

## 运行前准备

准备经过质量检查的数据和固定 manifest：

```bash
python scripts/download_data.py
garbage prepare-data --set data.data_dir data/raw --strict
```

运行产生的文件已经保留了大部分必要信息：

- 解析后的 `config.yaml` 与 `metrics.csv`；
- 源压缩包校验和与数据质量补丁版本；
- `data/manifests/summary.txt` 中的 manifest 校验和；
- 依赖锁、随机种子和选中的 checkpoint。

用验证集指标比较配置。只有选定最终配置后，才在测试集运行 `garbage evaluate`。兼容的
完成运行可直接排序，无需手动检查 checkpoint：

```bash
garbage compare-runs artifacts/experiment-a artifacts/experiment-b \
  --metric macro_f1 --output artifacts/comparison.csv
```

该命令会拒绝 `manifest_identity` 不同的运行。

## 一次具体的思考过程

[ResNet50 参考运行](../recorded-run/README.zh-CN.md)适合作为对照，因为 manifest、配置、
环境和输出都可以查看。假设想回答的问题是：**MixUp 是否改善了这组设置的验证集表现？**

- 保持 manifest、seed、ResNet50、15 个 epoch、optimizer、scheduler 和预处理不变。
- 只改变 `train.mixup_alpha`，从 `0.2` 改为 `0.0`。
- 对照运行在第 15 个 epoch 得到 `0.8976` validation balanced accuracy 和 `0.8858`
  macro F1。新的对比结果应读取它自己的 `metrics.csv`，不要使用测试集分数来选择设置。
- 对照运行中，`trash` 的测试样本只有 14 张，recall 最低，为 `0.7143`。除了汇总指标，
  还值得看看这个类别的验证表现和具体错误是否发生变化。
- 每个设置只运行一次，只能提示这次发生了什么，不能证明 MixUp 的普遍规律。换一个 seed
  或数据集，结论都可能变化。

这样的顺序能把问题、保持不变的设置、唯一改动、观察结果和限制连在一起。后面的短对比
也可以按同样方式理解。

## 可选对比 1：完整训练流程

这次运行建立一个起点，同时跑通完整训练、评测和错误分析流程。

```bash
garbage train \
  --config configs/imbalance_none.yaml \
  --set model.name resnet50 \
  --set train.epochs 15 \
  --set train.batch_size 32 \
  --set train.lr 0.001 \
  --set train.mixup_alpha 0.2 \
  --set data.num_workers 0 \
  --set run_name exp-resnet50-mixup

garbage evaluate \
  --checkpoint artifacts/exp-resnet50-mixup/best.pt \
  --plot
```

可以比较 accuracy、balanced accuracy、macro F1、每类 recall 和主要混淆类别。这个数据集
类别不平衡，因此 balanced accuracy 和 macro F1 通常比 accuracy 更有解释力。

## 可选对比 2：预训练与从零训练

使用相同的 MobileNetV3 配置，只改变是否加载预训练权重：

```bash
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set model.pretrained true \
  --set train.epochs 15 --set train.mixup_alpha 0.2 \
  --set data.num_workers 0 --set run_name exp-pretrained

garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set model.pretrained false \
  --set train.epochs 15 --set train.mixup_alpha 0.2 \
  --set data.num_workers 0 --set run_name exp-scratch
```

比较收敛速度和最佳验证集 balanced accuracy。对于小数据集，预训练权重通常是更好的
起点；从零训练则能直观看出迁移学习带来的价值。

## 可选对比 3：类别不平衡策略

三份配置只在类别不平衡处理方式上不同：

```bash
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set run_name exp-imbalance-none
garbage train --config configs/imbalance_weighted_loss.yaml \
  --set model.name mobilenetv3_small_100 --set run_name exp-imbalance-loss
garbage train --config configs/imbalance_weighted_sampler.yaml \
  --set model.name mobilenetv3_small_100 --set run_name exp-imbalance-sampler
```

比较验证集 balanced accuracy、macro F1，以及 `trash` 类的 precision/recall。加权和
过采样常在 precision 与 recall 之间取舍，并不存在适合所有部署目标的唯一方案。

## 可选对比 4：MixUp 与 EMA

以一份配置为基础，每次只改变一个选项：

```bash
# 参考组
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set train.mixup_alpha 0 \
  --set train.ema false --set run_name exp-regularization-none

# MixUp
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set train.mixup_alpha 0.2 \
  --set train.ema false --set run_name exp-regularization-mixup

# EMA
garbage train --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 --set train.mixup_alpha 0 \
  --set train.ema true --set train.ema_decay 0.99 \
  --set run_name exp-regularization-ema
```

用 `python scripts/plot_metrics.py <metrics.csv>` 绘制曲线。MixUp 常会提高训练损失，
但改善验证集泛化；EMA 需要足够多的优化步数才能跟上训练权重，短训练需谨慎解读。

## 可选对比 5：测试时增强

选定一个 checkpoint 后，对比普通推理与 TTA：

```bash
garbage evaluate --checkpoint artifacts/<run>/best.pt
garbage evaluate --checkpoint artifacts/<run>/best.pt --tta
```

TTA 会平均原图和水平翻转图的预测。它需要额外一次推理，因此可以一起查看指标变化和
推理耗时，再判断它是否值得使用。

后续分析工具：

```bash
python scripts/plot_metrics.py artifacts/<run>/metrics.csv
garbage explain --checkpoint artifacts/<run>/best.pt \
  --image data/raw/paper/paper1.jpg --output gradcam.png
```
