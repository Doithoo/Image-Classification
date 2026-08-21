# Choosing Models

[简体中文](choosing-models.zh-CN.md) | [Model reference](../reference/model-zoo.md)

Start with `learning_minimal.yaml` and ResNet-18 when learning the pipeline.
Use `resnet50.yaml` as the practical pretrained CNN baseline. MobileNetV3 and
RegNet variants suit constrained latency or memory; ConvNeXt and EfficientNet
provide additional CNN comparisons. Swin and ViT require more data and careful
compute budgeting.

Hold the manifest, seed, epochs, optimizer and preprocessing fixed while
comparing architectures. Select by validation metric, then evaluate the one
selected checkpoint on test. `garbage list-models` reports static provider,
input-size and Grad-CAM support without downloading weights.
