# Troubleshooting

[简体中文](troubleshooting.zh-CN.md) | [Documentation home](../README.md)

## `verify-data` fails

Do not bypass the check. A manifest checksum mismatch means a split file was
modified; a source checksum mismatch means image bytes, paths or class folders
changed. Review the change, then recreate manifests with `prepare-data
--overwrite`.

## Device or pretrained-weight failure

Use `--device cpu` for a local functional check. Explicit `cuda` and `mps`
requests fail when unavailable. Pretrained weights may require network access;
use `model.pretrained: false` when learning offline.

## Existing output directory

Runs, manifests, ONNX files, prediction JSON and evaluation directories reject
silent replacement. Choose a new name, safely resume the same run with
`last.pt`, or pass the narrow command's explicit `--overwrite` only after
reviewing the old evidence.

## Checkpoint incompatibility

Schema-v2 checkpoints load with PyTorch tensor-only mode. A resume also checks
model, class order, preprocessing and manifest identity. Recreate a model from
the source run rather than disabling those checks. Legacy checkpoints may be
used for prediction only when their tensor-only payload can be read safely.

## Memory pressure

Lower `train.batch_size`, use a smaller registered model, reduce input size in
a controlled experiment, and retain validation/test manifests. Record the
change under a new run name.
