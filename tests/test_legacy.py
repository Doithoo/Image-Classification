"""Legacy hand-written model migration: registry coverage + forward consistency.

The legacy models are vendored from the original ``Code/model`` tree and exposed
as ``legacy_*`` registry keys. They have no pretrained weights, so the checks
here are structural: every legacy entry builds and produces the same output
shape as its maintained timm counterpart on the same input.
"""

import warnings

import torch

from garbage_classifier.models import available_models, create_model

warnings.filterwarnings("ignore")

# legacy name -> timm counterpart (both must output (1, C) for the same input)
_LEGACY_TIM_MAP = {
    "legacy_resnet18": "resnet18",
    "legacy_resnet50": "resnet50",
    "legacy_mobilenet_v3_small": "mobilenetv3_small_100",
    "legacy_efficientnet_b0": "efficientnet_b0",
    "legacy_efficientnetv2_s": "efficientnetv2_s",
    "legacy_convnext_tiny": "convnext_tiny",
    "legacy_swin_tiny": "swin_tiny_patch4_window7_224",
    "legacy_vit_base_patch16_224": "vit_base_patch16_224",
}


def test_all_legacy_models_build_and_forward():
    x = torch.randn(1, 3, 224, 224)
    legacy = [n for n in available_models() if n.startswith("legacy_")]
    assert len(legacy) >= 16  # at least the original families
    for name in legacy:
        net = create_model(name, num_classes=6, pretrained=False).eval()
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 6), f"{name} produced {out.shape}"


def test_legacy_output_shape_matches_timm():
    """Same input -> same output shape as the maintained counterpart."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    for legacy_name, timm_name in _LEGACY_TIM_MAP.items():
        legacy = create_model(legacy_name, num_classes=6, pretrained=False).eval()
        timm_model = create_model(timm_name, num_classes=6, pretrained=False).eval()
        with torch.no_grad():
            a = legacy(x)
            b = timm_model(x)
        assert a.shape == b.shape == (1, 6), f"{legacy_name} vs {timm_name}"


def test_registry_does_not_import_legacy_models_eagerly():
    """Importing the models package must not touch legacy code (lazy loading)."""
    import subprocess
    import sys

    code = (
        "import sys; "
        "from garbage_classifier.models import available_models; "
        "assert 'garbage_classifier.models.legacy_models' not in sys.modules; "
        "assert any(n.startswith('legacy_') for n in available_models()); "
        "print('lazy-ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="src" if False else ".")
    assert out.returncode == 0, out.stderr
    assert "lazy-ok" in out.stdout
