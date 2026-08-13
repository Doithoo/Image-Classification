# 教程 5：推理与部署

> 目标：把训练得到的 checkpoint 变成可运行的预测工具，理解演示和生产部署的区别。

## 1. 单张图片

```bash
garbage predict --checkpoint artifacts/<run>/best.pt \
  --image data/raw/paper/paper1.jpg --top-k 3
```

checkpoint 中已经保存模型名称、类别名、输入尺寸和归一化参数，因此推理不需要
手工复制训练配置。输出包含 top-k 类别和概率。

## 2. 批量预测

```bash
garbage predict --checkpoint artifacts/<run>/best.pt \
  --image data/raw/paper --top-k 3 > predictions.txt
```

传入目录时，命令会按文件名排序处理支持的图片。需要结构化结果时，可在 Python 中
调用 `Predictor.predict_path`，把结果写入 CSV。

## 3. TTA 与 Grad-CAM

```bash
garbage predict --checkpoint artifacts/<run>/best.pt \
  --image data/raw/paper/paper1.jpg --tta
garbage explain --checkpoint artifacts/<run>/best.pt \
  --image data/raw/paper/paper1.jpg --output artifacts/gradcam.png
```

TTA 多做一次水平翻转推理，可能提升稳定性但会增加延迟。Grad-CAM 是分析工具，
用于检查模型关注区域，不是准确率保证。

## 4. 导出 ONNX

```bash
uv pip install -e '.[onnx]'
garbage export-onnx --checkpoint artifacts/<run>/best.pt \
  --output artifacts/<run>/model.onnx
```

导出会同时生成 `.onnx.meta.yaml`，记录输入尺寸和类别。ONNX 模型适合没有 Python
训练环境的轻量推理服务；`onnxruntime` 默认使用 CPU，也可以在支持的环境中配置 GPU
执行提供者。导出后的模型应使用一张已知图片与 PyTorch 结果做抽样比对。

## 5. Gradio 演示

```bash
uv pip install -e '.[demo]'
garbage demo --checkpoint artifacts/<run>/best.pt
```

这适合本地学习和展示。`--share` 会创建临时公开链接，不等于生产部署。

## 6. 从演示到生产的差距

生产系统还需要：

- 固定模型和类别版本，记录输入预处理；
- 请求校验、超时、日志和错误处理；
- 并发控制、资源限制和健康检查；
- 模型更新、回滚和输入数据监控。

本项目提供的是可复现的模型文件和推理入口，帮助你学习这些组件的边界；它不是
已经托管的在线服务。
