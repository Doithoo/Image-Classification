"""ONNX export: convert a self-contained checkpoint into a deployable model.

The checkpoint carries the full config (architecture, input size, class names),
so export never requires repeating model or preprocessing details.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..models.registry import create_model
from ..training.checkpoint import deployable_model_state, load_checkpoint, restore_config_from_checkpoint


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    image_size: int | None = None,
    opset: int = 17,
    verify: bool = True,
    device: str = "cpu",
) -> Path:
    """Export model weights from a checkpoint to ONNX; returns the output path."""
    import importlib.util

    if importlib.util.find_spec("onnx") is None:
        raise SystemExit("onnx is not installed; run: pip install -e '.[onnx]'")
    ckpt = Path(checkpoint_path)
    out = Path(output_path)
    payload = load_checkpoint(ckpt)
    cfg = restore_config_from_checkpoint(payload)
    if image_size is None:
        image_size = cfg.data.image_size

    model = create_model(cfg.model.name, num_classes=len(payload["class_names"]), pretrained=False)
    model.load_state_dict(deployable_model_state(payload))
    model.eval()

    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    # dynamo=False: use the stable TorchScript-based exporter (no onnxscript dependency)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )

    if verify:
        _verify_onnx(out, dummy, num_classes=len(payload["class_names"]))

    # write a sidecar with runtime metadata for serving
    meta = out.with_suffix(".onnx.meta.yaml")
    meta.write_text(
        f"model: {cfg.model.name}\nimage_size: {image_size}\nclasses: {payload['class_names']}\nopset: {opset}\n"
    )
    return out


def _verify_onnx(path: Path, dummy: torch.Tensor, num_classes: int) -> None:
    """Sanity-check the exported graph with onnxruntime if available."""
    try:
        import numpy as np  # noqa: F401 (used by onnxruntime input below)
        import onnxruntime as ort
    except ImportError:
        return  # verification skipped when onnxruntime is not installed

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: dummy.numpy()})[0]
    assert out.shape == (1, num_classes), f"unexpected ONNX output shape: {out.shape}"
    print(f"ONNX verified: input {inp} -> output {out.shape}")
