# Scripts

[简体中文](README.zh-CN.md)

This directory contains small, task-oriented utilities. They support the main
workflow but are not part of the installed `garbage` command.

| File | Purpose | When to use it |
|---|---|---|
| `download_data.py` | Download, checksum and prepare the public dataset | Once when setting up data |
| `preview_dataset.py` | Create a class contact sheet and count table | Before training or after changing data |
| `plot_metrics.py` | Plot loss and metric curves from `metrics.csv` | After a training run |

Use the three code areas like this:

```text
examples/   short programs to read and modify while learning
scripts/    supporting utilities you run for a specific task
src/        the installed application and reusable library
```

Every script supports `--help`. Start with:

```bash
python scripts/preview_dataset.py --help
```

Generated `__pycache__/` directories belong to Python and can be ignored.
