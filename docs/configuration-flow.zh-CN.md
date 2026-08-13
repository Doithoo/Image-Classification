# 配置与参数传递

每次训练只使用一份解析后的 `ExperimentConfig`。参数不会散落在全局变量中，也不会在
训练和推理代码里各复制一份。

## 覆盖优先级

越靠后的来源优先级越高：

```text
config.py 默认值
  < YAML 配置
  < --set 点号覆盖
  < 专用命令行参数（--device、--output-dir）
```

例如：

```bash
garbage show-config --config configs/resnet50.yaml \
  --set train.lr 0.0001 \
  --set train.batch_size 16 \
  --device cpu
```

`show-config` 与 `train` 使用完全相同的解析和校验流程，只是不加载数据、不创建模型。
当你不确定哪个参数真正生效时，先运行它。

## 参数流向哪里

| 配置区块 | 主要使用者 | 例子 |
|---|---|---|
| `data` | transforms、manifest、DataLoader | 图片尺寸、归一化、切分、workers |
| `model` | 模型注册表 | 模型名、预训练、类别数 |
| `train` | Trainer 与训练辅助模块 | epochs、优化器、学习率、MixUp、EMA |
| 顶层 | 命令编排 | device、输出目录、run name |

最终配置保存为 `artifacts/<run>/config.yaml`。checkpoint 同时保存模型、类别和预处理
元数据，确保推理与训练一致。

## 两类函数参数

- 工作流函数接收配置对象，例如 `train_from_config(cfg)`。
- 小型算法接收明确参数，例如 `confusion_matrix(labels, preds, num_classes)`。

这样既能集中管理实验设置，又能让独立算法容易理解和测试。
