# ONNX、Demo 与解释

[English](onnx-and-demo.md) | [教程](../tutorial/05-evaluation-and-deployment.zh-CN.md)

`export-onnx` 仅依据 checkpoint 元数据重建内置模型，并写入包含类别、预处理、checkpoint hash 和 opset 的 YAML sidecar；安装 ONNX Runtime 时还会比较 ONNX 与 PyTorch logits。已有输出必须显式使用 `--overwrite`。使用外部 factory 的 checkpoint 还必须传入 `--config <已审查的训练配置>`。

`demo` 是可选功能，需要安装 `.[demo]`。它适合本地检查，不是带认证的生产服务。外部 factory 同样要求显式提供已审查配置。

`explain` 仅对标记为支持的注册 CNN 使用 Grad-CAM。结果是模型注意区域的证据，不是因果解释的证明；应与高置信错误和每类指标一起审阅。
