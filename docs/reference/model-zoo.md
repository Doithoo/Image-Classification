# Model Zoo

[简体中文](model-zoo.zh-CN.md) | [Choosing models](../guides/choosing-models.md)

`garbage list-models` is the authoritative runtime listing. It reports provider,
provider-specific default input size and Grad-CAM support without constructing a
model or downloading weights. For timm entries, `ModelSpec` is populated from
timm's static pretrained configuration, including crop-derived resize size,
normalization and interpolation; TorchVision's ResNet-50 uses its V2 weight
transform contract.

Built-in entries come from `timm` except `tv_resnet50`, which is the
TorchVision comparison entry. Every entry has a `ModelSpec` recording provider,
upstream name, ImageNet preprocessing defaults and explanation support. The
registry replaces the final classifier for the prepared number of classes.

External model factories are explicit `module:function` values in
`model.factory`. They are trusted local code, so their source, dependencies and
parameters are part of experiment provenance. Inference, evaluation, export and
the demo require the reviewed training config to be supplied explicitly before
an external factory can be imported. Never approve a factory path from an
unreviewed checkpoint or config source.
