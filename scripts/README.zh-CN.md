# 辅助脚本

[English](README.md)

这里放的是任务单一的小工具。它们辅助正式工作流，但不属于安装后的 `garbage` 命令。

| 文件 | 用途 | 什么时候运行 |
|---|---|---|
| `download_data.py` | 下载、校验并准备公开数据集 | 第一次准备数据 |
| `preview_dataset.py` | 生成类别样本总览和数量表 | 训练前或修改数据后 |
| `plot_metrics.py` | 根据 `metrics.csv` 绘制训练曲线 | 训练完成后 |

```text
examples/   用来阅读和修改的短小教学程序
scripts/    为特定任务运行的辅助工具
src/        安装后的正式应用与可复用代码
```

每个脚本都支持 `--help`，例如 `python scripts/preview_dataset.py --help`。
Python 自动生成的 `__pycache__/` 可以忽略。
