# 教程 0：深度学习最小概念

> 目标：先建立一张能读懂日志和代码的地图，不要求数学推导。

## 1. 一次预测发生了什么

```text
图片文件
  ↓ 读取、裁剪、归一化
Tensor: [3, 224, 224]
  ↓ model(images)
logits: [6]                 每个类别的原始得分
  ↓ softmax
概率: [6]，总和为 1
  ↓ argmax
预测类别: cardboard / glass / ...
```

`logits` 还不是概率；训练时通常把 logits 直接交给交叉熵损失函数。

## 2. 训练如何改变模型

```text
预测 logits + 真实标签
        ↓ loss（错得有多远）
        ↓ backward（计算每个参数的梯度）
        ↓ optimizer.step（沿梯度更新参数）
```

- **参数**：模型需要学习的数字。
- **batch**：一次送入模型的一小组图片。
- **epoch**：完整看过一遍训练集。
- **learning rate**：每次更新迈多大步。
- **validation**：训练过程中用来选择设置的数据。
- **test**：最后一次检查泛化能力的数据。

## 3. 第一次训练建议

先使用关闭高级技巧的配置：

```bash
garbage train --config configs/learning_minimal.yaml \
  --set train.epochs 2 --set run_name basics-first-run
```

看到 `train_loss`、`val_loss` 和 `macro_f1` 变化后，再进入完整的
[训练策略教程](04-training.zh-CN.md)。每次只打开一个功能：预训练、MixUp、EMA、
类别重加权。

## 4. 自测

1. 为什么 `logits` 不需要先手动 softmax 才能计算交叉熵？
2. 如果训练集 loss 下降而验证集 loss 上升，这说明什么？
3. 为什么测试集不应该用来挑选学习率？
