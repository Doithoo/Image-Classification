# 教程 4：训练策略选择指南

> 适用读者：能跑通训练，但面对一堆超参数（优化器、学习率、batch size、epochs、
> 增强、不平衡处理）不知道"为什么选这个、不选那个"。
> 读完你会：拿到一个新任务，能按一套可解释的流程定出第一版配置；遇到
> 训练曲线异常，知道该调什么。

## 0. 一个总原则

**先跑通，再调优。** 第一版配置的目标不是"最好"，而是"能在合理时间内得到
一条看起来健康的训练曲线"。本文给出的默认值都是经验起点，不是真理。

---

## 1. 优化器怎么选

| 优化器 | 适用场景 | 默认学习率 |
|---|---|---|
| **AdamW**（本项目默认） | 绝大多数情况；预训练微调首选 | 1e-3（微调 1e-4~5e-4） |
| SGD + momentum | 从零训练经典 CNN；老派但稳定 | 1e-2~1e-1（需配合更大学习率） |
| Lion | 实验性；某些任务收敛快 | 1e-3~3e-3 |

**怎么选（新手版）**：
- 用预训练权重微调 → **AdamW**，lr 1e-4~5e-4（教程 2 说的迁移学习场景）。
- 从零训练大网络 → 可以试 SGD，但 AdamW 也完全没问题，别纠结。
- **别在优化器上花太多时间**：优化器之间的差距通常小于"学习率选没选对"。

## 2. 学习率：最重要的超参数

### 2.1 量级速查表

| 场景 | 学习率起点 |
|---|---|
| 预训练微调（AdamW） | 1e-4 ~ 5e-4 |
| 从零训练（AdamW） | 1e-3 |
| 从零训练（SGD） | 1e-2 |
| 分类头重新初始化、backbone 冻结 | 1e-3（只训 head） |

### 2.2 学习率与 batch size 的关系

**经验法则：batch size 翻倍 → 学习率也翻倍**（线性缩放）。

原因：梯度是多个样本的平均，batch 越大梯度越"稳定"，可以迈更大的步子。
所以教程 1 里说"OOM 时 batch size 减半，学习率也减半" —— 不是玄学，是这个法则。

### 2.3 怎么判断学习率不对

| 曲线特征 | 诊断 | 处理 |
|---|---|---|
| loss 完全不降 | lr 太小或结构问题 | lr × 10 试 |
| loss 先降后 NaN/暴涨 | lr 太大 | lr ÷ 10 |
| loss 快速降到很低但验证集差 | 过拟合（见 §6） | 增强/正则化，不是调 lr |

**专业技巧（lr finder 思想）**：小范围扫 lr，看 loss 曲线：
```bash
for lr in 1e-5 1e-4 1e-3 1e-2; do
  garbage train --config configs/resnet50.yaml --set train.lr $lr \
    --set train.epochs 3 --set data.num_workers 0 --set run_name lr-$lr
done
# 比较四条的 val_loss：选"降得最快又不抖"的那个量级
```

### 2.4 学习率调度（warmup + cosine 是什么）

```bash
garbage train --config configs/resnet50.yaml --set train.epochs 30
# 默认: warmup 5 epoch 线性从 1% 爬到 100%，再 cosine 退火到 ~0
```

- **warmup**：开局学习率从 1% 爬升。为什么？预训练模型微调时，新初始化的
  分类头还在乱跳，一开始就用满 lr 容易把 backbone 冲坏。
- **cosine 退火**：后期学习率平滑降到接近 0，让权重稳定落位。
- 新手默认 `warmup_epochs: 5`、`scheduler: cosine` 就好，不用改。

## 3. batch size：受硬件约束的选项

| 设备 | 推荐 batch size（224 输入, resnet50） |
|---|---|
| CPU / 8GB 显存 | 16–32 |
| 16GB 显存（T4） | 32–64 |
| 24GB+ 显存 | 64–128 |

- **OOM**：batch size ÷ 2，**学习率 ÷ 2**（§2.2 法则）。
- batch 太小（<16）梯度噪声大；太大（>128）对小数据集收益递减。
- 本项目默认 32，对练习足够。

## 4. epochs：训练多久

| 场景 | epochs 建议 |
|---|---|
| 预训练微调（小数据） | 15–30 + 早停 |
| 从零训练 | 30–60 + 早停 |
| 追求上限 | 60–100 + 早停 |

- **早停**（默认开启，`early_stop_patience: 15`）：验证集指标连续 N 个 epoch
  没创新高就停。别傻跑满 epochs。
- 判断"该不该继续"：看验证集指标（不是训练 loss）是否还在涨。平台期超过
  patience 就停。

## 5. 数据增强：什么时候开什么

| 增强 | 默认 | 什么时候加强 |
|---|---|---|
| RandomResizedCrop + Flip（基础） | 开 | 永远开 |
| RandAugment | 配置 `data.aug: randaug` | 数据少/容易过拟合时 |
| MixUp / CutMix | `train.mixup_alpha: 0.2` | 过拟合明显时（train loss 低、val 差） |
| Label Smoothing | 0.1 | 一直开着就行 |

**MixUp 的典型信号**：打开后训练 loss 变高、验证集变好 —— 这是正常的！
MixUp 生成的样本本身"更难"，但它让模型学特征而非死记。

## 6. 三大失败模式与诊断流程

```
训练曲线异常？
├── train_loss 不降
│   ├── lr 太小？ → lr × 10
│   └── 数据/模型没对上？ → 先 --dry-run 检查
├── train_loss 降，val_loss 也降 → 健康，继续
└── train_loss 降，val 不降/变差（过拟合）
    ├── 数据量小？ → 预训练 + 增强（randaug/mixup）+ 早停
    ├── epochs 太多？ → 早停已自动处理
    └── 模型太大？ → 换小模型（mobilenet）
```

**关键指标判断**：
- `train_loss` 低、`val_loss` 高 → **过拟合**（记忆训练集，泛化差）
- 两者都高 → **欠拟合**（容量不够/学得不够/学习率问题）
- 两者都低 → 健康，可以加大训练时间搏上限

## 7. 类别不平衡怎么办（本项目的核心问题）

数据里 `trash` 只有 5.4% 训练样本。三种策略对比（见 `configs/imbalance_*.yaml`）：

| 策略 | 做法 | 代价 |
|---|---|---|
| 不处理 | 普通 CE loss | 稀有类被牺牲（recall 低） |
| weighted loss | loss 里按类别频率反比加权 | 提升 recall，但 precision 下降（误报多） |
| weighted sampler | 采样时过采样稀有类 | 同上，trade-off 略不同 |

**选择建议**：
- 如果你的目标指标是 **balanced accuracy / macro F1**（每类平等）→ 考虑加权。
- 如果部署场景**误报成本高**（把金属当垃圾不如漏掉）→ 别加权，保留高 precision。
- **先跑"不处理"基线**，看稀有类到底差多少，再决定要不要加权 —— 别一上来就加权。
- 三组配置应使用同一份 manifest 和同一组训练参数，只改变类别不平衡策略；
  根据验证集选择方案，最后只评测一次测试集。

## 8. 一份"第一版配置"的决策流程（拿来即用）

```
新任务来了
  │
  ├─ 数据 < 1 万张？ ──────────── 是 → pretrained: true（迁移学习）
  │                                  └ AdamW, lr 1e-4, epochs 20, 早停 10
  │
  ├─ 数据 ≥ 1 万张？ ──────────── 是 → 可考虑从零，或继续预训练
  │
  ├─ 稀有类占比 < 10%？ ───────── 是 → 评测指标用 balanced_accuracy / macro_f1
  │                                   先跑 none 基线，再对比加权
  │
  ├─ 硬件内存小？ ─────────────── 是 → batch 16-32，lr 按 §2.2 缩放
  │
  └─ 输出：第一版配置 → --dry-run → 小训练（2-5 epoch）验证曲线健康
       → 完整训练 + 早停 → evaluate 测试集 → 记录到 docs/experiments.zh-CN.md
```

## 9. 练习

```bash
# 练习 1：lr 扫描（§2.3 的命令），记录四条 val_loss 曲线
# 练习 2：为每类复制最多 8 张图，生成真实的小数据 manifest 后体验过拟合
rm -rf /tmp/garbage-overfit
for class_dir in data/raw/*; do
  class=$(basename "$class_dir"); mkdir -p "/tmp/garbage-overfit/$class"
  find "$class_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    | head -8 | xargs -I{} cp "{}" "/tmp/garbage-overfit/$class/"
done
garbage prepare-data --set data.data_dir /tmp/garbage-overfit \
  --set data.manifest_dir /tmp/garbage-overfit-manifests
garbage train --config configs/imbalance_none.yaml --set model.name mobilenetv3_small_100 \
  --set data.data_dir /tmp/garbage-overfit \
  --set data.manifest_dir /tmp/garbage-overfit-manifests \
  --set train.epochs 30 --set data.num_workers 0 --set train.mixup_alpha 0 \
  --set run_name overfit-demo
# 练习 3：打开 mixup 0.2 重跑，对比两条 val 曲线 —— 用图片说明"正则化"的含义
```

**学完本系列**：你现在能回答"数据→模型→训练→评测"每个环节的选择题。回到
[学习路线](../learning-path.zh-CN.md) 的阶段 7，完成你自己的实验记录和毕业项目。
