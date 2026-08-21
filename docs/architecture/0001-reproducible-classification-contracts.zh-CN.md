# ADR 0001：可复现分类契约

[English](0001-reproducible-classification-contracts.md)

- 状态：已接受
- 范围：准备数据、模型构建、checkpoint、运行和评测证据

## 决策

准备数据拥有 schema-v2 identity，由有序类别、切分策略、全部 CSV checksum、每类数量和源图像字节共同派生。训练和评测在声明指标前都校验 identity。

checkpoint 是 tensor-only、带 schema 的原子写入产物。恢复训练校验模型规格、有序类别、预处理和数据 identity。旧 checkpoint 不会被静默升级为可恢复运行。

运行和评测目录默认不可变。新运行必须使用空目标；评测使用 split/TTA 专属路径；替换只能通过范围明确的 `--overwrite`。

模型注册表将静态 `ModelSpec` 元数据与构建分离。发现命令不下载权重。外部 factory 是显式可信 import，并记录为溯源信息。

## 影响

准备和校验增加 I/O，但本地数据改变后结果仍可检查。checkpoint 和证据包含更多元数据，但预测、导出和比较不依赖未记录的 YAML 状态。这些契约提升可复现性，但不承诺不同硬件或库版本间的数值完全一致。
