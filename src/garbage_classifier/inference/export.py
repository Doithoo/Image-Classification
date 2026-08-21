"""ONNX export with checkpoint-owned preprocessing metadata and numerical verification."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml

from ..models.registry import create_model
from ..training.checkpoint import (
    deployable_model_state,
    load_checkpoint,
    validate_inference_model_source,
)
from ..utils import file_sha256, pick_device, write_text_atomic


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    image_size: int | None = None,
    opset: int = 17,
    verify: bool = True,
    device: str = "cpu",
    overwrite: bool = False,
    config_path: str | Path | None = None,
) -> Path:
    """Export a checkpoint and optionally compare ONNXRuntime logits with PyTorch."""
    if importlib.util.find_spec("onnx") is None:
        raise SystemExit("onnx is not installed; run: pip install -e '.[onnx]'")
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"ONNX output already exists: {output}; use --overwrite")
    payload = load_checkpoint(checkpoint)
    cfg = validate_inference_model_source(payload, config_path)
    size = image_size or cfg.data.image_size
    resolved_device = pick_device(device)
    model = create_model(
        cfg.model.name,
        num_classes=len(payload["class_names"]),
        pretrained=False,
        factory=cfg.model.factory,
        params=cfg.model.params,
    )
    model.load_state_dict(deployable_model_state(payload))
    model.to(resolved_device).eval()
    dummy = torch.randn(1, 3, size, size, device=resolved_device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    if verify:
        _verify_onnx(output, model, dummy)
    metadata = {
        "schema_version": 2,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "model": payload.get("model"),
        "preprocessing": payload.get("preprocessing"),
        "classes": payload["class_names"],
        "input_name": "input",
        "output_name": "logits",
        "image_size": size,
        "opset": opset,
    }
    write_text_atomic(output.with_suffix(".onnx.meta.yaml"), yaml.safe_dump(metadata, sort_keys=False))
    return output


def _verify_onnx(path: Path, model: torch.nn.Module, dummy: torch.Tensor) -> None:
    """Verify output shape and numeric closeness when ONNX Runtime is available."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    with torch.inference_mode():
        reference = model(dummy).detach().cpu().numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    actual = session.run(None, {name: dummy.detach().cpu().numpy()})[0]
    if actual.shape != reference.shape:
        raise ValueError(f"unexpected ONNX output shape: {actual.shape}; expected {reference.shape}")
    maximum_error = float(abs(actual - reference).max())
    if not torch.allclose(torch.from_numpy(actual), torch.from_numpy(reference), rtol=1e-3, atol=1e-4):
        raise ValueError(f"ONNX output diverges from PyTorch (max absolute error {maximum_error:.3e})")
    print(f"ONNX verified: max absolute logit error {maximum_error:.3e}")
