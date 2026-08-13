# 教程系列（Tutorials）

> 面向普通初学者的**实操教程**，回答"具体怎么做"：不同设备怎么跑、数据怎么备、
> 模型怎么建、训练策略怎么选。
>
> 与[学习路线](../learning-path.zh-CN.md)的关系：学习路线是"按阶段做什么"的
> 课程表；本系列是"每一步的完整细节"的参考书。建议先跟着学习路线走，卡住时
> 翻对应教程。

| 教程 | 内容 | 适合谁 |
|---|---|---|
| [教程 0：深度学习最小概念](00-basics.zh-CN.md) | 图片、标签、logits、概率、loss、梯度、batch、epoch | 第一次接触深度学习的人 |
| [教程 1：设备与环境](01-environment.zh-CN.md) | CPU / NVIDIA GPU / Apple 芯片 / Kaggle 四种环境：装什么、怎么验证、速度预期、常见坑 | 第一次跑训练的人 |
| [教程 2：数据准备深潜](02-data.zh-CN.md) | 数据目录结构、manifest 生成、切分原理、用自己的数据、transforms 逐项讲解 | 想搞懂数据管道的人 |
| [教程 3：模型构建与定制](03-model.zh-CN.md) | backbone + 分类头心智模型、注册表、换模型、从零手写 CNN 并注册 | 想理解模型的人 |
| [教程 4：训练策略选择](04-training.zh-CN.md) | 优化器/学习率/batch size/epochs/增强/不平衡 的选择逻辑、三大失败模式诊断、第一版配置决策流程 | 能跑通但不会调参的人 |
| [教程 5：推理与部署](05-inference-deployment.zh-CN.md) | 单图、批量、TTA、Grad-CAM、ONNX、Gradio 和部署边界 | 想把模型用起来的人 |

## 快速路线

1. **先理解术语** → [教程 0](00-basics.zh-CN.md)（15 分钟）
2. **没跑起来** → [教程 1](01-environment.zh-CN.md)（装环境 + dry-run）
3. **数据哪来的** → [教程 2](02-data.zh-CN.md)
4. **模型怎么换** → [教程 3](03-model.zh-CN.md)
5. **超参怎么定** → [教程 4](04-training.zh-CN.md)
6. **模型怎么交付** → [教程 5](05-inference-deployment.zh-CN.md)
7. **都学会了** → 回到[学习路线](../learning-path.zh-CN.md)做毕业项目
