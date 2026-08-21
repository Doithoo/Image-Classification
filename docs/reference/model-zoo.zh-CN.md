# 模型库

[English](model-zoo.md) | [模型选择](../guides/choosing-models.zh-CN.md)

`garbage list-models` 是权威的运行时列表。它读取 provider 对应的默认输入尺寸和 Grad-CAM 支持情况，不会构造模型或下载权重。timm 条目的 `ModelSpec` 来自其静态预训练配置，包含由 crop ratio 推导的 resize 尺寸、归一化和插值；TorchVision ResNet-50 使用 V2 权重的变换契约。

内置条目除 `tv_resnet50`（TorchVision 对照）外来自 `timm`。每个条目都有 `ModelSpec`，记录 provider、上游名称、ImageNet 预处理默认值和解释支持。registry 会将末端分类头替换为准备后类别数。

外部模型 factory 使用 `model.factory` 中显式的 `module:function`。它们是可信本地代码，其源、依赖和参数都属于实验溯源的一部分。推理、评测、导出和 demo 在导入外部 factory 前，都要求显式提供已审查的训练配置。不要批准未经审查的 checkpoint 或配置中的 factory 路径。
