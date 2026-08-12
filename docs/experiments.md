# Experiments（实验记录）

> 定位：学习项目 —— 这些实验用于演示完整工作流（训练 → 评测 → 出图 → 对比），
> 全部命令可直接复现。设备：Apple Silicon (MPS)。

## 实验 1：ResNet50 从零训练（15 epochs, MixUp）

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
```

**测试集结果**（`garbage evaluate --checkpoint artifacts/exp-resnet50-mixup/best.pt --plot`）：

| 类别 | Precision | Recall | F1 | support |
|---|---:|---:|---:|---:|
| cardboard | 0.951 | 0.951 | 0.951 | 41 |
| glass | 0.904 | 0.922 | 0.913 | 51 |
| metal | 0.857 | 0.878 | 0.867 | 41 |
| paper | 0.950 | 0.950 | 0.950 | 60 |
| plastic | 0.915 | 0.878 | 0.896 | 49 |
| trash | 0.714 | 0.714 | 0.714 | 14 |
| **accuracy** | | | **0.906** | 256 |
| **balanced acc** | | | **0.882** | |
| **macro F1** | | | **0.882** | |

15 epochs 的从零训练就超过了历史 README 中 300 epochs 的 resnet50（0.871 acc）——
MixUp + randaug + cosine + 更好的超参是主要差异。

## 实验 2：MobileNetV3-Small 从零训练（15 epochs, MixUp）

```bash
garbage train \
  --config configs/imbalance_none.yaml \
  --set model.name mobilenetv3_small_100 \
  --set train.epochs 15 \
  --set train.batch_size 32 \
  --set train.lr 0.001 \
  --set train.mixup_alpha 0.2 \
  --set data.num_workers 0 \
  --set run_name exp-mobilenet-mixup
```

**测试集结果**：accuracy **0.781**，balanced acc **0.750**，macro F1 **0.767**
（trash 类 F1 0.696）。参数量只有 resnet50 的 1/15，精度差距 ~12 个点 ——
典型的"容量换速度"权衡。

## 对比结论（学习要点）

| 模型 | Params | test acc | balanced acc | macro F1 | 训练耗时 (MPS) |
|---|---:|---:|---:|---:|---:|
| resnet50 | 23.5M | 0.906 | 0.882 | 0.882 | 25 min |
| mobilenetv3_small | 1.5M | 0.781 | 0.750 | 0.767 | 6 min |

1. **trash 类始终最差**（F1 0.71/0.70，样本仅 5.4%）—— 这就是类别不平衡的直接体现，
   也解释了为什么必须看 balanced acc / macro F1 而非只看 accuracy。
2. **MixUp 的效果**：训练 loss 偏高（软标签使然）但验证集稳步上升，是正则化的典型特征。
3. 下一步练习建议：跑 `configs/imbalance_weighted_loss.yaml` 对比 trash 类 F1 是否改善；
   或把 `mixup_alpha` 调成 0 对比有无 MixUp 的差异；或对 resnet50 做预训练微调（
   `pretrained: true`）看精度的上限在哪。

## 产物约定

每次训练在 `artifacts/<run>/` 留下：`config.yaml`（完整配置）、`metrics.csv`（逐 epoch
指标）、`best.pt` / `last.pt`（自包含 checkpoint）、评测时的 `predictions.csv` /
`errors.csv` / `confusion_matrix.png`，以及可选的 `loss_curve.png`
（`python scripts/plot_metrics.py artifacts/<run>/metrics.csv`）。

---

# 补充实验（v1.2）

## 实验 3：类别不平衡三策略对比（MobileNetV3-Small，15 epochs，无 MixUp）

三个配置仅在重平衡策略上不同（`configs/imbalance_*.yaml`），其余完全一致，
best checkpoint 按验证集 balanced accuracy 选择。

| 策略 | valid bal_acc | test acc | test balanced | trash P | trash R | trash F1 |
|---|---:|---:|---:|---:|---:|---:|
| none（基线） | 0.818 | **0.832** | **0.815** | 0.833 | 0.714 | **0.769** |
| inverse loss | 0.836 | 0.812 | 0.790 | 0.600 | 0.643 | 0.621 |
| weighted sampler | **0.837** | 0.816 | 0.812 | 0.647 | **0.786** | 0.710 |

**学习要点（反直觉但真实）**：
1. 重平衡在**验证集**上赢了（0.836/0.837 vs 0.818），但在**测试集**上基线反而最好 ——
   基于验证集选模型不保证迁移到测试集，样本越小波动越大。
2. inverse loss 把 trash 的精度压到 0.60（为了提 recall 而误报过多），trash F1 反而
   从 0.769 掉到 0.621 —— 重平衡不是免费的，它是"精度↔召回"的权衡。
3. weighted sampler 确实提升了 trash 召回（0.714→0.786），代价是整体 accuracy 略降。

## 实验 4：从零训练 vs 预训练微调（ResNet50）

| 训练方式 | epochs | lr | test acc | test balanced | 备注 |
|---|---:|---:|---:|---:|---|
| 从零 + MixUp | 15 | 1e-3 | **0.906** | **0.882** | 见实验 1 |
| ImageNet 预训练微调 | 10 | 3e-4 | 0.883 | 0.853 | TTA 后 acc 0.887 |

**学习要点**：预训练**不一定**更好。垃圾照片与 ImageNet 类别分布差异大（域偏移），
小数据集上从零训练 + 强增强（MixUp + randaug）反而追平甚至反超。真实项目里
"先试从零 + 强增强，再试预训练微调"是合理的对比思路。

## 实验 5：TTA（测试时增强）效果

对 exp-resnet50-mixup（从零 resnet50）在测试集对比：

| 方式 | test acc | test balanced |
|---|---:|---:|
| 普通推理 | 0.906 | 0.882 |
| TTA（水平翻转平均） | **0.922** | **0.905** |

**学习要点**：TTA 只花一倍推理时间，白捡 1-2 个点 —— 平均多个增强视图的概率
输出，降低单次决策的方差。`evaluate --tta` / `predict --tta` 随时可用。

## Grad-CAM 使用示例

```bash
garbage explain --checkpoint artifacts/exp-resnet50-mixup/best.pt \
  --image data/raw/paper/paper1.jpg --output gradcam.png
# 输出: top prediction: paper (class 3) + 热力图叠加图
```

看"模型看哪里"是错误分析的第一步：比如 trash 误判成 metal 时，热力图会告诉你
模型是看中了金属反光还是形状。
