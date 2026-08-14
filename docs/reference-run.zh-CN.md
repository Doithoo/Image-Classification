# 经过核对的 ResNet50 参考运行

## 为什么保留这次运行

这次运行提供的是一份可以检查的结果，而不是一个需要照抄的分数。配置、训练曲线、
选中的 checkpoint、测试指标和类别错误都来自同一次完整运行，方便初学者把评测流程和
真实产物对应起来。

## 数据集与固定切分

经过检查的示例数据集共有 2,524 张图片，包含六个类别。严格切分使用 seed `666` 和
`0.8/0.1/0.1` 比例，得到 2,017 张训练图片、252 张验证图片和 **255 张留出测试图片**。
`data/manifests/summary.txt` 记录的训练 manifest 校验值是
`4ad59108d1d0d216`。

这个校验值用于识别本次结果对应的 manifest。即使模型配置相同，只要下载的数据、切分
或 manifest 不同，结果就可能不同。

## 准确配置

本次运行使用[公开的 ResNet50 配置](../configs/reference_resnet50.yaml)：训练 15 个 epoch，
batch size 为 32，采用 AdamW、带 5 个 warmup epoch 的 cosine scheduler、`0.1` label
smoothing、`0.2` MixUp、RandAugment 和预训练 ResNet50。`best.pt` 由 balanced
accuracy 选出。

## 记录下来的运行环境

训练生成的 `run.yaml` 记录了：

- 开始时间 `2026-08-14T07:33:20.319748+00:00`；
- 耗时 `1636.6` 秒，约 27 分 17 秒；
- Python `3.11.15`、PyTorch `2.13.0`；
- `macOS-26.5.2-arm64-arm-64bit`，设备 `mps`，accelerator `arm64`；
- Git revision `dc2b22a`。

revision 是理解运行背景的一部分。之后修正的默认数据路径不会改变这次运行所用的
manifest 和模型输入。

## 验证集结果

第 15 个 epoch 的 validation balanced accuracy 最高，为 `0.8976`；同时 validation
accuracy 为 `0.8929`，macro F1 为 `0.8858`，validation loss 为 `0.6352`。曲线显示
validation loss 总体下降，但中间有几次正常回升，并不是一条完美平滑的改善轨迹。

![训练与验证曲线](assets/reference-resnet50-curves.png)

## 留出测试集结果

使用 `best.pt` 评测后，`evaluation.json` 记录了以下汇总指标：

| 指标 | 数值 |
| --- | ---: |
| Accuracy | `0.9333` |
| Balanced accuracy | `0.9079` |
| Macro F1 | `0.9176` |
| Weighted F1 | `0.9325` |

这里需要区分两件事：验证集指标用于选择 checkpoint；255 张测试图片只在之后用于报告
最终结果。

## 各类别透露了什么

glass 的每类 F1 最高，为 `0.9709`，测试样本为 51 张。样本最少的 `trash` recall 最低，
为 `0.7143`，F1 为 `0.8000`：14 张图片中有 10 张预测正确，其余 4 张分别被预测为
cardboard、metal、paper 和 plastic。测试集只有 14 张 trash 图片，一次错误就会让该类
recall 变化超过 7 个百分点，因此更适合继续查看具体样本，不适合得出很确定的普遍结论。

![测试集混淆矩阵](assets/reference-resnet50-confusion.png)

## 这份结果能说明什么

这份证据说明仓库可以完成一次 ResNet50 训练与评测，并保留足够的元数据供检查。它只适用
于这份数据 manifest、这组配置、这个代码 revision 和这台机器。它不是 ResNet50 的通用
基准，也不表示每个类别都已经解决，更不保证其他设备或数据集会得到相同分数。

## 重新运行

```bash
python scripts/download_data.py
garbage prepare-data --strict --set data.data_dir data/raw
garbage train --config configs/reference_resnet50.yaml
garbage evaluate --config configs/reference_resnet50.yaml \
  --checkpoint artifacts/reference-resnet50/best.pt \
  --output-dir artifacts/reference-resnet50/evaluation --plot
python scripts/plot_metrics.py artifacts/reference-resnet50/metrics.csv \
  --out artifacts/reference-resnet50/reference-resnet50-curves.png
```

可以把新生成的 `run.yaml`、`metrics.csv` 和 `evaluation.json` 与上面的数值逐项比较。
即使 seed 相同，不同硬件和底层库实现仍可能带来很小的数值差异。
