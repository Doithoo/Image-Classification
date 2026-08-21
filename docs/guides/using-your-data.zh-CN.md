# 换成自己的图片数据

这份指南把同一套流程用于另一个按文件夹分类的数据集。开始前需要先安装项目，并确保
每张图片只有一个类别标签。

## 整理目录

每个类别使用一个文件夹，图片直接放在类别文件夹下：

```text
my-dataset/
├── apples/
│   ├── apple-001.jpg
│   └── apple-002.jpg
├── bananas/
│   └── banana-001.jpg
└── oranges/
    └── orange-001.jpg
```

支持 `.jpg`、`.jpeg` 和 `.png`。扫描器只读取类别文件夹这一层，嵌套更深的图片不会
进入 manifest。

## 切分前先看图片

长训练之前先生成预览：

```bash
python scripts/preview_dataset.py --data-dir my-dataset
```

打开生成的预览图，确认文件夹名称与图片内容一致。放错目录的图片和含义模糊的类别，
越早发现越容易处理。

## 使用单独的 manifest

不要覆盖垃圾分类示例的 manifest，为这份数据指定独立目录：

```bash
garbage prepare-data --strict \
  --set data.data_dir my-dataset \
  --set data.manifest_dir data/my-manifests
```

`--strict` 会在发现内容完全相同的图片时停止。先检查报出的路径和标签，再决定保留哪一
份，不要直接忽略重复数据。

接着查看两份元数据：

```bash
cat data/my-manifests/summary.txt
cat data/my-manifests/source.yaml
```

确认识别出的类别、每类数量、切分大小和数据根目录。类别名称和
`model.num_classes` 来自 `source.yaml`，不需要在模型配置中手工填写类别数。

## 先检查一个 batch

第一次使用最小配置：

```bash
garbage train --config configs/learning_minimal.yaml --dry-run \
  --set data.data_dir my-dataset \
  --set data.manifest_dir data/my-manifests \
  --set run_name my-data-check
```

出现 `dry-run OK`，说明 manifest、变换、标签、模型输出和反向传播能够连起来运行。
这一步不评价模型效果。

## 运行一个短基线

先保持设置简单，不要同时改变多个参数：

```bash
garbage train --config configs/learning_minimal.yaml \
  --set data.data_dir my-dataset \
  --set data.manifest_dir data/my-manifests \
  --set train.epochs 3 \
  --set run_name my-data-baseline
```

三个 epoch 可以生成初始曲线和 checkpoint，适合检查数据与流程，但不代表模型已经达到
有竞争力的结果。

评测时继续使用同一份数据与 manifest：

```bash
garbage evaluate --checkpoint artifacts/my-data-baseline/best.pt \
  --set data.data_dir my-dataset \
  --set data.manifest_dir data/my-manifests \
  --plot
```

比较 accuracy、balanced accuracy 和 macro F1，再看每类 recall、混淆矩阵和
`errors.csv`。这些现象比立即更换模型更适合作为下一步的依据。

## 常见问题

| 现象 | 常见原因 | 先检查什么 |
|---|---|---|
| 没有找到图片 | 图片嵌套过深或扩展名不支持 | 确认结构是 `my-dataset/<类别>/<图片>` |
| unreadable image | 文件损坏或下载不完整 | 单独打开报错图片 |
| strict 模式报告重复 | 相同内容出现多次 | 比较报出的路径和标签 |
| 某个切分缺少类别 | 该类别图片太少 | 先补充数据，不急着调模型 |
| accuracy 尚可但 macro F1 很低 | 类别不平衡或某类被忽略 | 看类别数量和每类 recall |
| 训练很好但验证很差 | 过拟合或切分存在问题 | 检查预览、重复数据和增强方式 |

理解短基线后，每次只改变一个因素。[实验说明](experiments.zh-CN.md)展示了怎样保持
对比结果容易解释。
