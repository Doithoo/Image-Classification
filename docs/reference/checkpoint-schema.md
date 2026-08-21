# Checkpoint Schema

[简体中文](checkpoint-schema.zh-CN.md) | [Training tutorial](../tutorial/04-training.md)

New checkpoints use schema version 2 and are saved through a same-directory
temporary file followed by `os.replace`. They are loaded with
`torch.load(weights_only=True)`; checkpoint payloads never contain executable
model objects.

| Field | Purpose |
| --- | --- |
| `schema_version` | compatibility contract, currently `2` |
| `model`, `model_state_dict` | model factory specification and deployable tensors |
| `training_model_state_dict` | fast weights needed by resume |
| `config`, `preprocessing` | resolved settings and exact input contract |
| `class_names`, `manifest_identity` | ordered labels and prepared-data binding |
| optimizer/scheduler/EMA/scaler | optimization continuation state |
| `rng_state` | tensor-only Python, NumPy, Torch and CUDA random state |
| `extra` | last validation metrics and future non-executable metadata |

Prediction reconstructs only the built-in model contract in the checkpoint.
External factories are executable trusted code and therefore require an
explicit matching reviewed config before import. Resume requires `last.pt` and
matching model specification, class order, preprocessing and manifest identity;
it also reconciles `metrics.csv` with the completed checkpoint epoch.
Schema-less legacy tensor-only checkpoints can be read for limited prediction
compatibility, but cannot resume schema-v2 training.
