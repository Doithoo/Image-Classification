# 分类流程

[English](classification-flow.md) | [教程](../tutorial/README.zh-CN.md)

本文沿着一张图像讲解准备、训练、评测与部署。第一次实际运行请使用[教程](../tutorial/README.zh-CN.md)。

## 1. 数据流总览

```
GitHub Release (garbage-classification.tar.gz)
        │  scripts/download_data.py  (SHA-256 校验 + 解压)
        ▼
data/raw/  (class 文件夹)
        │  garbage prepare-data  (分层切分 seed=666 + 坏图/重复图校验)
        ▼
data/manifests/{train,valid,test}.csv   ← 可移植清单（相对路径）
        │  garbage train  (读取 manifest + transform)
        ▼
artifacts/<run>/  (config.yaml + metrics.csv + best.pt + last.pt)
        │  garbage evaluate / predict / export-onnx / explain
        ▼
测试集指标 / 单图预测 / ONNX 模型 / Grad-CAM 热力图
```

## 2. 为什么用 CSV manifest 而不是目录遍历

- **可移植**：manifest 存的是相对路径（正斜杠），换机器、换目录都能用，
  不依赖特定操作系统或运行目录。
- **可复现**：manifest 由固定种子（666）分层切分生成，并记录 `dataset.yaml`
  （数据根目录、有序类别和完整切分/源数据 identity）。验证后的同一数据与同一
  manifest identity 才是同一份数据契约。

## 3. 训练循环里发生了什么（按 epoch）

```
for epoch in range(epochs):
    for batch in train_loader:
        images, labels = batch
        if mixup/cutmix enabled:  images, soft_labels = mixup(images, labels)
        outputs = model(images)
        loss = soft_cross_entropy(outputs, soft_labels)  # 或 CrossEntropyLoss
        loss.backward(); optimizer.step()
    lr_scheduler.step()          # warmup → cosine
    if ema: ema.update(model)    # 影子权重
    在验证集上评测 → 记录 metrics.csv → 更新 best.pt
    若 patience 个 epoch 无提升 → early stop
```

### 3.1 Warmup（学习率预热）
训练一开始就把学习率拉满，容易让网络"冲过头"。warmup 用线性从 1% 爬到目标
学习率，再交给 cosine 退火。`SequentialLR` 把两个调度器串起来。

### 3.2 MixUp / CutMix（软标签数据增强）
MixUp 混合两张完整图片；CutMix 把一张图的矩形区域替换成另一张图的块。两者都按
有效图像面积比例混合标签，因此损失函数必须接受软标签。实现说明见
`training/mixup.py`。

### 3.3 EMA（权重指数滑动平均）
见 `training/ema.py` 顶部的原理说明。验证和保存 best 用影子权重，训练继续用
快权重 —— 影子权重更平滑、泛化更好。

配置 EMA 时要注意两个细节：
1. **BN 的 running_mean/var 也要纳入平均**：只平均权重会导致验证时影子权重配
   fast 模型的 BN 统计，微调阶段统计失配 → 激活饱和 → 预测坍缩成单一类别。
2. **decay 的时间常数是 1/(1−decay) 步**：0.999 = 1000 步。15 epoch（~945 步）
   的训练里 0.999 的影子根本追不上，反而拖后腿。EMA 是给长训练用的。

### 3.4 类别不平衡三策略
`configs/imbalance_*.yaml` 提供三种可比配置：
- `none`：普通 CE loss（基线，trash 会被牺牲）
- `inverse`：loss 里按类别频率反比加权（`training/weights.py`）
- `weighted`：采样时过采样稀有类（`WeightedRandomSampler`）

评测指标选 `balanced_accuracy` / `macro_f1` 而不是 accuracy —— 因为 accuracy
会被多数类主导。

## 4. Checkpoint 为什么自包含

`training/checkpoint.py` 保存：权重 + 优化器/调度器状态 + 类别表 + 预处理参数
+ 完整配置 + git revision + 随机数状态。

**好处**：`garbage predict --checkpoint xxx.pt` 不需要你手动告诉它"类别是什么、
均值方差是多少、模型叫什么" —— 全部从 checkpoint 恢复，杜绝训练/推理配置漂移。
这能避免推理配置漂移。完整复现实验还需要完全一致的 manifest、数据质量补丁版本与
依赖锁文件。

## 5. 评测为什么报这么多指标

测试集只有 256 张，trash 只有 14 张。只看 accuracy 的话，一个"永远猜 paper"的
模型也能拿 23% —— 所以 `evaluate` 还输出 balanced accuracy、macro/weighted F1、
每类 precision/recall/F1、混淆矩阵，以及 predictions.csv / errors.csv
（错误样本清单，用于人工检查模型错在哪）。

## 5.5 Grad-CAM 与 TTA（推理期工具）

- **Grad-CAM**（`garbage explain`）：取最后一个卷积层的特征图，用类别得分对它的
  梯度作为通道权重，加权求和后得到"哪些像素支撑了这个决策"的热力图。
  用途：错误分析 —— 模型分错时，看它看的是不是错误区域。原理见
  `inference/gradcam.py` 顶部注释。
- **TTA**（`evaluate --tta` / `predict --tta`）：推理时把输入水平翻转再推一次，
  两个 softmax 取平均；实验 5 用来比较指标收益和额外推理成本。

## 6. 推荐的学习路线

1. 跑通一次完整训练：`garbage train --config configs/resnet50.yaml`（先
   `--set train.epochs 5` 快速看流程）
2. 用 `scripts/plot_metrics.py` 画出 loss 曲线，观察过拟合/欠拟合
3. 对比三个 `imbalance_*.yaml`，看 trash 类的 F1 如何变化
4. 打开 `mixup_alpha: 0.2`，观察训练 loss 变高但验证集变好（正则化的典型特征）
5. 读 `training/` 下四个文件的注释，每个都是经典论文的一页笔记
