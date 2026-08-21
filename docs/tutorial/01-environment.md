# Environment

[Previous: concepts](00-basics.md) | [Next: data](02-data.md)

Use Python 3.10, 3.11 or 3.12. `uv` reads the committed lock file so a new
environment receives the dependency versions used by CI.

```bash
uv sync --locked --extra dev
uv run garbage --version
uv run garbage list-models
uv run garbage show-config --config configs/learning_minimal.yaml --sources
```

`show-config` does not load images or construct a model. It is the right first
command when a YAML value and a command-line override appear to disagree.

CUDA and MPS are optional. `--device auto` chooses CUDA, then MPS, then CPU;
requesting an unavailable explicit device fails during preflight.
