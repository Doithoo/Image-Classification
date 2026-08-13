# Tests

[简体中文](README.zh-CN.md)

Tests are primarily for contributors and CI. Beginners do not need to read them
before using or studying the project, but selected tests are useful executable
examples of expected behavior.

## Suggested reading order

1. `test_examples.py`: the beginner examples are runnable.
2. `test_metrics.py`: small inputs make metric formulas concrete.
3. `test_end_to_end.py`: training, evaluation, prediction and export work together.

## Source map

| Test | Main code under test |
|---|---|
| `test_config.py`, `test_cli.py` | `config.py`, `cli.py` |
| `test_manifest.py` | `data/manifest.py` |
| `test_metrics.py`, `test_evaluation.py` | `evaluation/` |
| `test_training_techniques.py` | MixUp, CutMix, EMA, schedulers |
| `test_checkpoint_resume.py` | Checkpoint save and resume |
| `test_explain_tta.py` | Grad-CAM and TTA |
| `test_packaging.py` | Package metadata, CI and documentation links |

Run one focused test while learning:

```bash
pytest tests/test_metrics.py -v
```

Run all quality checks before contributing:

```bash
pytest
ruff check .
ruff format --check .
```

Generated `__pycache__/` and `.pytest_cache/` directories can be ignored.
