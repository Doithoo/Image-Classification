# 可复现实验手册

这份手册把项目变成一个小型图像分类实验室。每组实验只改变一个因素，在验证集上
选择 checkpoint，确定最终方案后只评测一次测试集。

## 运行前准备

准备经过质量检查的数据和固定 manifest：

```bash
python scripts/download_data.py
garbage prepare-data --set data.data_dir data/raw --strict
```

每次运行都保留：

- 解析后的 `config.yaml` 与 `metrics.csv`；
- 源压缩包校验和与数据质量补丁版本；
- `data/manifests/summary.txt` 中的 manifest 校验和；
- 依赖锁、随机种子和选中的 checkpoint。

用验证集指标比较配置。只有选定最终配置后，才在测试集运行 `garbage evaluate`。

## 实验 1：完整训练流程

这组实验建立参考结果，同时跑通完整训练、评测和报告流程。

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

记录 accuracy、balanced accuracy、macro F1、每类 recall 和主要混淆类别。这个数据集
类别不平衡，因此 balanced accuracy 和 macro F1 通常比 accuracy 更有解释力。

## 实验 2：预训练与从零训练

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

## 实验 3：类别不平衡策略

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

## 实验 4：MixUp 与 EMA

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

## 实验 5：测试时增强

选定一个 checkpoint 后，对比普通推理与 TTA：

```bash
garbage evaluate --checkpoint artifacts/<run>/best.pt
garbage evaluate --checkpoint artifacts/<run>/best.pt --tta
```

TTA 会平均原图和水平翻转图的预测。它需要额外一次推理，因此应同时记录指标变化和
推理耗时。

## 结果记录表

将自己的结果填入下表，不要直接比较来自不同 manifest 或环境的数字：

| 运行 | 改变因素 | 最佳 valid balanced acc | Test acc | Test balanced acc | Macro F1 | 备注 |
|---|---|---:|---:|---:|---:|---|
| 参考组 | 无 | | | | | |
| 实验组 | | | | | | |

后续分析工具：

```bash
python scripts/plot_metrics.py artifacts/<run>/metrics.csv
garbage explain --checkpoint artifacts/<run>/best.pt \
  --image data/raw/paper/paper1.jpg --output gradcam.png
```
