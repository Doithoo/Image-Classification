# 测试目录

[English](README.md)

测试主要服务贡献者和 CI。初学者使用项目前不需要通读，但部分测试是预期行为的
可执行示例。

## 推荐阅读顺序

1. `test_examples.py`：初学者示例是否可运行。
2. `test_metrics.py`：用小输入理解指标公式。
3. `test_end_to_end.py`：训练、评测、预测和导出的完整闭环。

| 测试 | 主要验证的源码 |
|---|---|
| `test_config.py`、`test_cli.py` | `config.py`、`cli.py` |
| `test_manifest.py` | `data/manifest.py` |
| `test_metrics.py`、`test_evaluation.py` | `evaluation/` |
| `test_training_techniques.py` | MixUp、CutMix、EMA、调度器 |
| `test_checkpoint_resume.py` | checkpoint 保存与恢复 |
| `test_packaging.py` | 包元数据、CI 和文档链接 |

学习时可运行 `pytest tests/test_metrics.py -v`。贡献前运行 `pytest`、`ruff check .`
和 `ruff format --check .`。自动生成的缓存目录可以忽略。
