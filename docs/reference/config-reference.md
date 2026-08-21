# Configuration Reference

[简体中文](config-reference.zh-CN.md) | [Configuration flow](../concepts/configuration-flow.md)

Resolution order is dataclass defaults, YAML, `--set KEY VALUE`, then dedicated
CLI flags. Inspect the result with `garbage show-config --sources`.

| Field | Default | Meaning |
| --- | --- | --- |
| `data.data_dir` | `data/raw` | class-folder source root; may override metadata root after identity verification |
| `data.manifest_dir` | `data/manifests` | prepared `dataset.yaml` and split CSV directory |
| `data.image_size` / `data.resize_size` | `224` / `256` | crop and resize sizes |
| `data.normalize_mean` / `data.normalize_std` | dataset values | RGB normalization used by fixed preprocessing |
| `data.split_ratios`, `data.seed` | `[.8,.1,.1]`, `666` | deterministic preparation policy |
| `data.num_workers`, `data.pin_memory` | `0`, `true` | DataLoader runtime settings |
| `data.aug` | `basic` | `basic` or `randaug` training augmentation |
| `data.interpolation` | `bilinear` | `bilinear` or `bicubic` resize interpolation |
| `model.name` | `resnet18` | registered model name |
| `model.num_classes` | `null` | optional assertion; runtime derives classes from metadata |
| `model.pretrained` | `false` | request provider pretrained weights |
| `model.preprocessing` | `fixed` | `fixed` project values or `model_default` registry ImageNet values |
| `model.params` | `{}` | explicit provider/model keyword parameters |
| `model.factory` | `null` | trusted extension `module:function` factory |
| `train.epochs`, `train.batch_size`, `train.lr`, `train.weight_decay`, `train.momentum`, `train.seed` | `20`, `32`, `.001`, `.0001`, `.9`, `42` | basic optimization controls |
| `train.optimizer`, `train.scheduler` | `adamw`, `cosine` | optimizer and schedule policy |
| `train.warmup_epochs` | `0` | must be less than epochs |
| `train.label_smoothing`, `train.mixup_alpha`, `train.cutmix_alpha` | `0` | regularization; MixUp and CutMix are mutually exclusive |
| `train.grad_clip`, `train.amp`, `train.ema`, `train.ema_decay` | `0`, `false`, `false`, `.999` | optional stability/performance tools |
| `train.early_stop_patience` | `5` | validation epochs without improvement; `0` disables stopping |
| `train.best_metric` | `macro_f1` | one of reported validation aggregate metrics |
| `train.class_weight`, `train.sampler` | `none`, `none` | imbalance alternatives |
| `device` | `auto` | `auto`, `cpu`, `cuda` or `mps` |
| `output_dir`, `run_name` | `artifacts`, `null` | isolated run destination |
| `log_level` | `info` | console logging threshold |

Classes are deliberately not configured here. Their ordered mapping is owned by
prepared data metadata and is embedded in every checkpoint.
