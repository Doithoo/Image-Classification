# Evaluation, Prediction And Deployment

[Previous: training](04-training.md) | [Tutorial home](README.md)

Evaluate the held-out split once after choosing settings with validation:

```bash
uv run garbage evaluate --checkpoint artifacts/baseline/best.pt \
  --set data.data_dir data/raw --plot
```

The command publishes `evaluation/test/` with `evaluation.json`, per-class
metrics, calibration data, all predictions and confidence-ranked errors.
Different split and TTA evaluations receive distinct directories.

```bash
uv run garbage predict --checkpoint artifacts/baseline/best.pt \
  --image data/raw/paper/paper1.jpg --output prediction.json
uv run garbage explain --checkpoint artifacts/baseline/best.pt \
  --image data/raw/paper/paper1.jpg --output gradcam.png
uv run garbage export-onnx --checkpoint artifacts/baseline/best.pt \
  --output model.onnx
```

Grad-CAM is limited to supported CNN specifications. ONNX export writes a
metadata sidecar and compares runtime logits when ONNX Runtime is installed.
