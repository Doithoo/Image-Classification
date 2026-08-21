# Training Configurations

[简体中文](README.zh-CN.md) | [Configuration reference](../docs/reference/config-reference.md)

Inspect final values before training:

```bash
uv run garbage show-config --config configs/learning_minimal.yaml --sources
```

| File | Use | Weights/network |
| --- | --- | --- |
| `learning_minimal.yaml` | first CPU dry run and transparent baseline | no pretrained download |
| `resnet50.yaml` | practical pretrained CNN baseline | ImageNet weight download/cache |
| `reference_resnet50.yaml` | settings for the recorded run | ImageNet weight download/cache |
| `swin_tiny.yaml` | transformer comparison | ImageNet weight download/cache |
| `imbalance_none.yaml` | imbalance baseline | ImageNet weight download/cache |
| `imbalance_weighted_loss.yaml` | inverse-frequency loss ablation | ImageNet weight download/cache |
| `imbalance_weighted_sampler.yaml` | weighted-sampler ablation | ImageNet weight download/cache |

Create a new `run_name` for every independent experiment. Test data is for the
final selected model; compare alternatives through validation evidence.
