# Class Imbalance

[简体中文](class-imbalance.zh-CN.md) | [Experiments](experiments.md)

The repository supplies three comparable configurations: no rebalancing,
inverse-frequency weighted loss, and weighted sampling. They are alternatives;
do not enable both loss weighting and sampling unless a documented experiment
requires it.

Use balanced accuracy and macro F1 to select a setting. Inspect per-class
precision and recall because an apparent aggregate improvement can hide a
large precision-recall tradeoff for rare classes. Keep the split and every
unrelated setting fixed.
