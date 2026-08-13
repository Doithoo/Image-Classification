# Experiments（历史记录，必须重跑）

> **旧基准已废弃**：下列数字全部来自未修补的上游 v1 数据，其中包含已复核的错误
> 标签与跨类别重复图片；历史流程还在比较技术时反复查看测试集。这些表只保留作为
> 历史来源，不能代表当前基准，也不能和 `v1.0-audit.1` 数据上的运行直接比较。
>
> 发布替代数字时，必须归档依赖锁、源压缩包校验和、补丁版本、生成的 manifest、
> 完整配置与 checkpoint。只在验证集选择模型，最终模型只评测一次测试集。完成重跑
> 前，本文不虚构任何新数字。

## 实验 1：ResNet50 预训练微调（15 epochs, MixUp）

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

TTA（`--tta`）后 accuracy 升到 0.922 / balanced 0.905。

## 实验 2：MobileNetV3-Small 从零 vs 预训练（15 epochs, MixUp）

| 训练方式 | valid bal | test acc | test balanced | 参数 |
|---|---:|---:|---:|---:|
| 从零（`pretrained: false`） | 0.492 | 0.551 | 0.479 | 1.5M |
| ImageNet 预训练微调 | 0.697 | 0.781 | 0.750 | 1.5M |

**学习要点**：预训练优势在小模型上非常明显（+27 个点的 balanced acc）—— 小模型
容量有限，从零学不动；但实验 1 里 resnet50 大模型从零+强增强也能追平预训练微调
（见实验 4 的教训：域偏移大时预训练不一定赢）。**结论：先试预训练微调，再对比
从零+强增强，用实验说话。**

## 实验 3：类别不平衡三策略对比（MobileNetV3-Small 预训练，15 epochs，无 MixUp）

三个配置仅在重平衡策略上不同（`configs/imbalance_*.yaml`），其余完全一致，
best checkpoint 按验证集 balanced accuracy 选择。

| 策略 | valid bal | test acc | test balanced | trash P | trash R | trash F1 |
|---|---:|---:|---:|---:|---:|---:|
| none（基线） | 0.818 | **0.832** | **0.815** | 0.833 | 0.714 | **0.769** |
| inverse loss | 0.836 | 0.812 | 0.790 | 0.600 | 0.643 | 0.621 |
| weighted sampler | **0.837** | 0.816 | 0.812 | 0.647 | **0.786** | 0.710 |

**历史观察（不能作为当前结论）**：
1. 重平衡在**验证集**上赢了，但在**测试集**上基线反而最好 —— 基于验证集选模型
   不保证迁移到测试集，样本越小波动越大。
2. inverse loss 把 trash 的精度压到 0.60，trash F1 反而从 0.769 掉到 0.621 ——
   重平衡是"精度↔召回"的权衡，不是免费的。
3. weighted sampler 确实提升 trash 召回（0.714→0.786），代价是整体 accuracy 略降。

## 实验 4：EMA 开关对比（MobileNetV3-Small 预训练，15 epochs, MixUp）

| 配置 | valid bal | test acc | test balanced | trash F1 |
|---|---:|---:|---:|---:|
| 无 EMA | 0.697 | **0.781** | **0.750** | **0.696** |
| EMA decay=0.99 | 0.695 | 0.777 | 0.731 | 0.571 |
| EMA decay=0.999 | 0.397 | 0.465 | 0.417 | 0.333 |

**学习要点（本实验踩了三个坑，都是真实教训）**：
1. **EMA 影子权重必须包含 BN 的 running_mean/var**。最初实现只平均权重、保留 fast
   模型的 BN 统计，结果验证集预测坍缩成"永远猜同一类"（balanced acc 恒为 1/6）。
   权重快速变化（微调阶段）时 BN 统计失配会饱和激活 —— 平均 running stats 后修复
   （timm 的 ModelEmaV2 也是这么做的）。
2. **decay 的时间常数 = 1/(1−decay) 步**：0.999 → 1000 步，0.99 → 100 步。15 epoch
   只有 945 步，0.999 的影子权重几乎没跟上训练，验证分数远低于 fast 模型。
3. **EMA 是为长训练设计的**：它平滑的是训练后期的权重抖动。短训练里 EMA 最多打平
   甚至拖后腿；想看到 EMA 收益，需要几十上百个 epoch（或更低的 decay）。

## 实验 5：TTA（测试时增强）效果

对 exp-resnet50-mixup 在测试集对比：

| 方式 | test acc | test balanced |
|---|---:|---:|
| 普通推理 | 0.906 | 0.882 |
| TTA（水平翻转平均） | **0.922** | **0.905** |

TTA 只花一倍推理时间，白捡 1.6 个点 —— 平均多个增强视图的概率输出，降低单次
决策的方差。`evaluate --tta` / `predict --tta` 随时可用。

## 工具示例

```bash
# Grad-CAM：看模型"看哪里"（错误分析第一步）
garbage explain --checkpoint artifacts/exp-resnet50-mixup/best.pt \
  --image data/raw/paper/paper1.jpg --output gradcam.png

# 训练前 1-batch 冒烟（验证数据/模型/设备没问题再跑长训练）
garbage train --config configs/resnet50.yaml --dry-run

# 模型参数/FLOPs 一览
garbage bench

# 画 loss 曲线
python scripts/plot_metrics.py artifacts/<run>/metrics.csv
```

## 产物约定

每次训练在 `artifacts/<run>/` 留下：`config.yaml`（完整配置，真相来源）、
`metrics.csv`（逐 epoch 指标）、`best.pt` / `last.pt`（自包含 checkpoint）、
评测时的 `predictions.csv` / `errors.csv` / `confusion_matrix.png`，以及可选的
`loss_curve.png`。
