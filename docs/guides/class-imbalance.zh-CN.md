# 类别不平衡

[English](class-imbalance.md) | [实验](experiments.zh-CN.md)

仓库提供三组可比较配置：不重平衡、逆频率加权损失和加权采样。它们是替代方案；除非有明确记录的实验目的，不应同时开启损失加权和采样。

使用 balanced accuracy 和 macro F1 选择设置，并检查每类 precision/recall。整体指标提高可能掩盖稀有类明显的精确率-召回率取舍。保持切分和其他设置不变。
