# 文档首页

[English](README.md) | [项目首页](../README.zh-CN.md)

按当前目标选择页面。教程按顺序阅读；指南和参考页适合按问题查阅。

## 我想跑通项目

1. [教程首页](tutorial/README.zh-CN.md)：从环境和一次 dry run 开始。
2. [使用自己的数据](guides/using-your-data.zh-CN.md)：创建隔离的 manifest。
3. [排障](guides/troubleshooting.zh-CN.md)：数据、设备、checkpoint 和输出目录问题。
4. [参考运行](recorded-run/README.zh-CN.md)：查看一次已完成运行的证据。

## 我想理解代码

- [分类流程](concepts/classification-flow.zh-CN.md)：从图像到指标证据。
- [代码导览](concepts/code-tour.zh-CN.md)：包边界和阅读顺序。
- [配置流](concepts/configuration-flow.zh-CN.md)：默认值、YAML 和 CLI 优先级。
- [架构决策](architecture/0001-reproducible-classification-contracts.zh-CN.md)：兼容性边界。

## 我需要具体答案

| 问题 | 页面 |
| --- | --- |
| 应修改哪个配置字段？ | [配置参考](reference/config-reference.zh-CN.md) |
| 数据和划分如何被标识？ | [数据格式](reference/dataset-format.zh-CN.md) |
| 指标和输出文件代表什么？ | [指标](reference/metrics.zh-CN.md) |
| checkpoint 中有什么？ | [checkpoint 协议](reference/checkpoint-schema.zh-CN.md) |
| 应从哪个模型开始？ | [模型选择](guides/choosing-models.zh-CN.md) |
| 如何比较实验？ | [实验指南](guides/experiments.zh-CN.md) |
| 类别不平衡策略有什么差别？ | [类别不平衡](guides/class-imbalance.zh-CN.md) |
| 如何部署或解释模型？ | [ONNX、Demo 与解释](guides/onnx-and-demo.zh-CN.md) |

配置和示例目录也有独立入口：[configs](../configs/README.zh-CN.md) 与 [examples](../examples/README.zh-CN.md)。
