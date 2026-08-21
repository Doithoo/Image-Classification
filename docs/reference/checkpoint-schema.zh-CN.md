# Checkpoint 协议

[English](checkpoint-schema.md) | [训练教程](../tutorial/04-training.zh-CN.md)

新 checkpoint 使用 schema version 2，先写同目录临时文件，再通过 `os.replace` 发布。加载使用 `torch.load(weights_only=True)`；payload 不包含可执行模型对象。

| 字段 | 用途 |
| --- | --- |
| `schema_version` | 当前为 `2` 的兼容性契约 |
| `model`, `model_state_dict` | 模型 factory 规格和部署张量 |
| `training_model_state_dict` | 恢复训练所需的快速权重 |
| `config`, `preprocessing` | 最终设置和准确输入契约 |
| `class_names`, `manifest_identity` | 有序标签和准备数据绑定 |
| optimizer/scheduler/EMA/scaler | 优化过程续跑状态 |
| `rng_state` | tensor-only 的 Python、NumPy、Torch 和 CUDA 随机状态 |
| `extra` | 最近验证指标和未来的不可执行元数据 |

预测只会直接重建 checkpoint 中的内置模型契约。外部 factory 是可执行可信代码，因此导入前必须显式提供内容匹配、已经审查的配置。恢复训练要求使用 `last.pt`，并校验模型规格、类别顺序、预处理和 manifest identity；同时会把 `metrics.csv` 与 checkpoint 已完成的 epoch 对齐。无 schema 的旧 tensor-only checkpoint 可有限用于预测，不能恢复 schema-v2 训练。
