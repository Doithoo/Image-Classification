# Recorded ResNet50 Run

[简体中文](README.zh-CN.md) | [Project home](../../README.md)

This directory separates the published result's explanation from its
machine-readable evidence. It does not contain a model checkpoint or source
images.

| Item | Value |
| --- | --- |
| Model | timm ResNet50, ImageNet initialized |
| Dataset | audited six-class garbage images, fixed strict split |
| Train / valid / test | 2,017 / 252 / 255 |
| Checkpoint selection | validation balanced accuracy |
| Test accuracy | 0.9333 |
| Test balanced accuracy | 0.9079 |
| Test macro F1 | 0.9176 |

`config.yaml` is the published requested configuration. **Historical schema-v1**
metadata and curves predate schema-v2 data and checkpoint contracts; they remain
evidence of that revision, not a schema-v2 artifact that can safely resume a
modern run. Reproduce a current reference by preparing new data and comparing
new data and comparing its generated `run.yaml`, `metrics.csv` and
`evaluation/evaluation.json` with the documented scope.

![Training curves](../assets/reference-resnet50-curves.png)

![Test confusion matrix](../assets/reference-resnet50-confusion.png)
