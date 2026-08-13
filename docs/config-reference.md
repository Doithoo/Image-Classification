# Configuration Reference

Use `garbage show-config` to inspect the values that will actually run. Defaults
below come from `config.py`; YAML and CLI overrides may replace them.

> **中文版：[config-reference.zh-CN.md](config-reference.zh-CN.md)**

## Data

| Key | Default | Meaning and guidance |
|---|---|---|
| `data.data_dir` | `data` | Class-folder root used by `prepare-data`; the download command commonly writes `data/raw`, so pass that explicitly. |
| `data.manifest_dir` | `data/manifests` | Location of split CSVs and `source.yaml`. Training reads from here. |
| `data.classes` | six garbage classes | Expected class names. Generated manifest metadata is authoritative for training. |
| `data.image_size` | `224` | Final square input. Larger values cost more memory/compute and affect ONNX shape. |
| `data.resize_size` | `256` | Resize before crop; must be at least `image_size`. |
| `data.normalize_mean` | `[0.673, 0.639, 0.604]` | RGB channel means used by train/eval/inference and saved in checkpoints. |
| `data.normalize_std` | `[0.208, 0.209, 0.231]` | Positive RGB standard deviations; must match the means' data source. |
| `data.split_ratios` | `[0.8, 0.1, 0.1]` | Train/valid/test ratios; three non-negative numbers summing to 1. |
| `data.seed` | `666` | Controls deterministic manifest splitting, not model training. |
| `data.num_workers` | `4` | Parallel data-loading processes. Use `0` for easiest debugging or constrained systems. |
| `data.pin_memory` | `true` | Speeds CPU-to-CUDA transfer; only applied on CUDA. |
| `data.aug` | `basic` | `basic` or `randaug`; start with basic, use RandAugment for stronger regularization. |

## Model

| Key | Default | Meaning and guidance |
|---|---|---|
| `model.name` | `resnet50` | Registry key. Run `garbage bench`; beginners can start with `resnet18` or MobileNetV3. |
| `model.num_classes` | `null` | Derived from manifest classes. Set only to assert an expected count. |
| `model.pretrained` | `true` | Load ImageNet weights. Usually preferred for small datasets; disable to study training from scratch. |

## Training

| Key | Default | Meaning and guidance |
|---|---|---|
| `train.epochs` | `60` | Maximum epochs. Use 2-5 for exercises; early stopping may finish sooner. |
| `train.batch_size` | `32` | Images per update. Halve it on OOM and reconsider the learning rate. |
| `train.lr` | `0.001` | Learning rate. Pretrained fine-tuning often starts at `0.0001`-`0.0005`. |
| `train.weight_decay` | `0.0001` | Parameter regularization; `0` disables it. |
| `train.momentum` | `0.9` | SGD momentum; ignored by AdamW/Lion. Must be in `[0, 1)`. |
| `train.optimizer` | `adamw` | `adamw`, `sgd`, or optional `lion`. AdamW is the beginner default. |
| `train.scheduler` | `cosine` | `cosine`, `step`, or `none`; updates LR after each epoch. |
| `train.warmup_epochs` | `5` | Linear warmup length. Use `0` for the minimal learning config. |
| `train.label_smoothing` | `0.1` | Softens hard labels; `[0, 1)`. Use `0` while learning the basic loss. |
| `train.mixup_alpha` | `0.0` | Positive value enables MixUp. It takes precedence over CutMix when both are positive. |
| `train.cutmix_alpha` | `0.0` | Positive value enables CutMix when MixUp is disabled. |
| `train.grad_clip` | `1.0` | Maximum gradient norm; `0` disables clipping. |
| `train.amp` | `true` | Mixed precision on CUDA/MPS. Disable when debugging numerical issues. |
| `train.early_stop_patience` | `15` | Stop after this many unimproved epochs; `0` stops after the first non-improvement. |
| `train.best_metric` | `macro_f1` | Metric used to select `best.pt`; valid metrics must exist in evaluation output. |
| `train.ema` | `false` | Maintain smoothed deployment weights. Learn the basic loop before enabling it. |
| `train.ema_decay` | `0.999` | EMA retention in `[0, 1)`; higher values need longer runs to catch up. |
| `train.class_weight` | `none` | `none`, `inverse`, or `effective` loss weighting for imbalance. |
| `train.sampler` | `none` | `none` or `weighted`; changes which training samples appear in batches. |
| `train.seed` | `42` | Controls model initialization, augmentation and training randomness. |

## Top Level

| Key | Default | Meaning and guidance |
|---|---|---|
| `device` | `auto` | `auto`, `cpu`, `cuda`, or `mps`. Dedicated `--device` has final precedence. |
| `output_dir` | `artifacts` | Parent directory for run outputs; `--output-dir` has final precedence. |
| `run_name` | `null` | Run folder name. If omitted, model name plus timestamp is used. |
| `log_level` | `info` | `debug`, `info`, `warning`, `error`, or `critical`. |

## Important Relationships

- `data.resize_size >= data.image_size`; changing image size changes memory,
  computation, preprocessing and exported ONNX input shape.
- `data.seed` controls data splitting; `train.seed` controls training randomness.
- Lowering `train.batch_size` solves memory pressure; LR often needs proportional
  adjustment when batch size changes substantially.
- `train.warmup_epochs` should be smaller than `train.epochs` for meaningful
  cosine training, although the validator permits any non-negative value.
- MixUp and CutMix create soft targets. When both alphas are positive, MixUp wins.
- Loss weighting and weighted sampling can be combined, but doing so may
  over-correct imbalance. Compare one change at a time.
- `model.num_classes: null` is safest: the manifest supplies the actual count.
- Checkpoint inference restores model and preprocessing metadata; it does not
  require repeating training configuration manually.

## CLI-Only Parameters

Some controls are command operations rather than experiment configuration:
`--resume`, `--dry-run`, `--checkpoint`, `--split`, `--plot`, `--tta`,
`--top-k`, `--image`, `--output`, `--opset`, `--no-verify`, `--class-idx`,
`--alpha`, `--share`, `--strict`, and `--legacy`. Run
`garbage <command> --help` for their accepted values and scope.
