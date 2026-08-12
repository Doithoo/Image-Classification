"""Legacy hand-written model implementations.

TODO(migration phase): the 16 hand-written models in ``Code/model/`` will be
wrapped here behind the registry (e.g. ``register("legacy_resnet50")``) after an
output-consistency check against the timm/torchvision counterparts. They stay
importable so old experiments remain reproducible, but are not the default.
"""
