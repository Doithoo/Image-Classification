# 实操教程

这里收录比学习路线更详细的操作说明。已经知道当前问题时，可以直接打开对应主题；
不确定阅读顺序时，先沿着[学习路线](../learning-path.zh-CN.md)完成一次短流程。

如果命令已经能运行，但不知道源码从哪里开始看，直接打开
[代码导览](../code-tour.zh-CN.md)。

| 教程 | 解决的问题 | 适合什么时候看 |
|---|---|---|
| [深度学习最小概念](00-basics.zh-CN.md) | 图片、标签、logits、概率、loss、梯度、batch、epoch 分别是什么 | 基础术语仍然陌生时 |
| [设备与环境](01-environment.zh-CN.md) | CPU、NVIDIA GPU、Apple 芯片和 Kaggle 应该怎样安装与检查 | 环境没有跑起来或速度异常时 |
| [数据准备](02-data.zh-CN.md) | 目录结构、manifest、切分、数据变换和常见数据问题 | 想看清图片怎样进入训练时 |
| [模型构建与定制](03-model.zh-CN.md) | backbone、分类头、模型注册表和自定义模型 | 想理解或替换模型时 |
| [训练策略选择](04-training.zh-CN.md) | 学习率、batch size、epoch、增强、不平衡和异常曲线 | 流程已跑通但不知道下一步改什么时 |
| [推理与部署](05-inference-deployment.zh-CN.md) | 单图、批量、TTA、Grad-CAM、ONNX 和 Gradio | 想使用或解释 checkpoint 时 |

遇到问题时，先回到最近生成的 `config.yaml`、`metrics.csv` 或 `errors.csv`。带着具体
现象查阅对应教程，通常比顺序读完所有细节更有效。

准备替换示例数据时，查看[换成自己的图片数据](../using-your-data.zh-CN.md)。
