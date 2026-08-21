# ONNX, Demo And Explanations

[简体中文](onnx-and-demo.zh-CN.md) | [Tutorial](../tutorial/05-evaluation-and-deployment.md)

`export-onnx` reconstructs a built-in model only from checkpoint metadata. It
writes a YAML sidecar containing classes, preprocessing, checkpoint hash and
opset; when ONNX Runtime is installed, it compares ONNX logits with PyTorch
logits. Existing outputs need explicit `--overwrite`. Checkpoints that use an
external factory additionally require `--config <reviewed-training-config>`.

`demo` is optional and requires `.[demo]`. It is a local inspection surface,
not an authenticated production service.

`explain` uses Grad-CAM only for registered CNNs marked as supported. The
result is evidence about model attention, not a proof of causal reasoning.
Always review it with confidence-ranked errors and class metrics.
