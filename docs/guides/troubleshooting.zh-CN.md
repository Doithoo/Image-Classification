# 排障

[English](troubleshooting.md) | [文档首页](../README.zh-CN.md)

## `verify-data` 失败

不要跳过校验。manifest checksum 不匹配表示切分文件被修改；source checksum 不匹配表示图像字节、路径或类别目录发生改变。检查变更后使用 `prepare-data --overwrite` 重新生成。

## 设备或预训练权重失败

本地功能检查可使用 `--device cpu`。显式请求不可用的 `cuda` 或 `mps` 会失败。预训练权重可能需要网络；离线学习可设 `model.pretrained: false`。

## 输出目录已存在

运行、manifest、ONNX、预测 JSON 和评测目录都拒绝静默替换。请使用新名称、用同一运行中的 `last.pt` 安全恢复，或确认旧证据不再需要后才使用对应命令的 `--overwrite`。

## checkpoint 不兼容

schema v2 checkpoint 使用 PyTorch tensor-only 模式加载。恢复训练还会检查模型、类别顺序、预处理和 manifest identity。不要关闭这些检查；应回到源运行重建。旧 checkpoint 只有在 tensor-only 可安全读取时才可用于预测。
