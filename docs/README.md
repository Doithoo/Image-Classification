# Documentation

[简体中文](README.zh-CN.md) | [Project home](../README.md)

Choose the page that matches the job in front of you. The tutorial is ordered;
guides and reference pages are designed for lookup.

## I Want To Run The Project

1. [Tutorial home](tutorial/README.md): start with environment and one dry run.
2. [Using your own data](guides/using-your-data.md): create isolated manifests.
3. [Troubleshooting](guides/troubleshooting.md): data, device, checkpoint and output errors.
4. [Recorded reference run](recorded-run/README.md): inspect a completed result.

## I Want To Understand The Code

- [Classification flow](concepts/classification-flow.md): image to metric evidence.
- [Code tour](concepts/code-tour.md): package ownership and reading order.
- [Configuration flow](concepts/configuration-flow.md): defaults, YAML and CLI precedence.
- [Architecture decision](architecture/0001-reproducible-classification-contracts.md): compatibility boundaries.

## I Need A Specific Answer

| Question | Page |
| --- | --- |
| Which field should I change? | [Configuration reference](reference/config-reference.md) |
| How are data and splits identified? | [Dataset format](reference/dataset-format.md) |
| What does a metric or output file mean? | [Metrics](reference/metrics.md) |
| What is inside a checkpoint? | [Checkpoint schema](reference/checkpoint-schema.md) |
| Which model should I start with? | [Choosing models](guides/choosing-models.md) |
| How should I compare experiments? | [Experiments](guides/experiments.md) |
| How do imbalance methods differ? | [Class imbalance](guides/class-imbalance.md) |
| How do I deploy or explain a model? | [ONNX, demo and explanations](guides/onnx-and-demo.md) |

The configuration and example directories have their own entry pages:
[configs](../configs/README.md) and [examples](../examples/README.md).
