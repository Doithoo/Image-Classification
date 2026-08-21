"""Demo web app: launch a Gradio interface on top of a checkpoint."""

from __future__ import annotations

from pathlib import Path

from .predictor import Predictor


def run_demo(
    checkpoint_path: str | Path,
    device: str = "auto",
    share: bool = False,
    config_path: str | Path | None = None,
) -> None:
    """Launch the Gradio UI. Requires the ``demo`` extra (gradio)."""
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("gradio is not installed; run: pip install -e '.[demo]'") from None

    predictor = Predictor(checkpoint_path, device=device, config_path=config_path)

    def classify(image):
        if image is None:
            return None
        from PIL import Image

        top = predictor.predict(Image.fromarray(image), top_k=3)
        return {name: prob for name, prob in top}

    gr.Interface(
        fn=classify,
        inputs=gr.Image(type="numpy", label="Upload a garbage photo"),
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        title="Garbage Image Classifier",
        description="Predicts cardboard / glass / metal / paper / plastic / trash.",
        allow_flagging="never",
    ).launch(share=share)
