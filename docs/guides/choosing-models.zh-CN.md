# 模型选择

[English](choosing-models.md) | [模型参考](../reference/model-zoo.zh-CN.md)

学习流程时从 `learning_minimal.yaml` 和 ResNet-18 开始。实际预训练 CNN 基线可使用 `resnet50.yaml`。MobileNetV3 与 RegNet 更适合受限的延迟或内存；ConvNeXt 与 EfficientNet 可作为其他 CNN 对照。Swin 和 ViT 通常需要更多数据和更谨慎的算力预算。

比较架构时固定 manifest、seed、epoch、优化器和预处理。用验证集指标选择，随后只对被选中的 checkpoint 评测测试集。`garbage list-models` 只读取静态规格，不会下载权重。
