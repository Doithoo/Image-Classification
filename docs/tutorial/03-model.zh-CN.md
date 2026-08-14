# 教程 3：模型构建与定制

> 适用读者：想理解"模型到底是怎么来的"，会换模型、加模型，甚至从零写一个 CNN。
> 读完你会：讲得清 backbone 和分类头的关系；能注册一个新模型；能写一个最简单的
> 卷积网络并跑通训练。

## 1. 模型的两层心智模型

图像分类模型 = **特征提取器（backbone）+ 分类头（head）**

```
输入图片 (3, 224, 224)
   ↓
[backbone 卷积网络]  →  把图片变成一张"特征图"，浓缩成 (D) 维向量
   ↓
[分类头 Linear(D, 6)]  →  映射到 6 个类别的得分（logits）
   ↓
softmax → 概率
```

- **backbone** 负责"看懂图片"：边缘 → 纹理 → 局部形状 → 整体语义。
- **head** 负责"投票"：把 backbone 理解到的特征归到 6 类之一。
- 预训练模型的 backbone 在 ImageNet（1000 类、1400 万张图）上已经学会"看懂
  通用图片"；这里只把 head 换成 6 类（`num_classes=6`），再在垃圾图片上微调。
  这就是迁移学习。

## 2. 本项目里模型怎么组织（注册表）

第一次阅读只看 `models/registry.py` 的 `register`、`create_model`，再看 `zoo.py`
如何注册和选择 timm 模型。

`garbage_classifier/models/registry.py` 维护一张"名字 → 工厂函数"的表：

```python
register("resnet50")(_timm_factory("resnet50"))   # 名字 -> 能造出模型的东西
```

配置里写 `model.name: resnet50`，训练时 `create_model("resnet50", num_classes=6,
pretrained=True)` 就会：
1. 从 timm 拿到 resnet50 骨架（ImageNet 预训练权重）
2. 把分类头换成 6 类的
3. 返回一个可直接 `model(images)` 的 `nn.Module`

**好处**：换模型 = 改配置里一个字段，代码零改动：

```bash
garbage train --config configs/resnet50.yaml --set model.name mobilenetv3_small_100
garbage train --config configs/resnet50.yaml --set model.name convnext_tiny
garbage train --config configs/resnet50.yaml --set model.name swin_tiny_patch4_window7_224
```

## 3. 怎么选 backbone（给新手的第一版建议）

先用 `garbage bench` 看每个模型的参数和 FLOPs，然后按场景选：

| 你的情况 | 建议 backbone | 理由 |
|---|---|---|
| 只想跑通流程、电脑慢 | `mobilenetv3_small_100`（1.5M） | 训练快 10 倍，精度够练习 |
| 默认主力 | `resnet50`（23.5M） | 经典均衡，文档实验都是它 |
| 想要更高精度 | `convnext_tiny`（27.8M） | 现代结构，精度好 |
| 想体验 Transformer | `swin_tiny_patch4_window7_224` | 与 CNN 完全不同的思路 |
| 移动端部署 | `mobilenetv3_small_100` / `efficientnet_b0` | 参数少、快 |

**参数量和 FLOPs 怎么看**：参数量大致影响模型文件和内存，FLOPs/MACs 反映一次
前向计算量。它们不是准确率保证；小模型可能更适合你的设备和延迟目标。第一次训练
建议按 `mobilenetv3_small_100 → resnet18 → resnet50` 的顺序体验，再尝试
`swin_tiny_patch4_window7_224` 这样的 Transformer。

## 4. 训练时模型参数怎么配置（ModelConfig）

`config.py` 里的 `ModelConfig` 三个字段：

| 字段 | 含义 | 新手建议 |
|---|---|---|
| `name` | 注册表键 | 见上表 |
| `num_classes` | 类别数 | 与你的数据类别数一致（本项目 6） |
| `pretrained` | 是否用 ImageNet 预训练权重 | 数据 < 1 万张强烈建议 `true` |

## 5. 如何添加一个"新模型"

### 5.1 如果它存在于 timm（最简单）

```bash
python -c "import timm; print([m for m in timm.list_models() if '你要的关键词' in m])"
```

找到名字后，改 `src/garbage_classifier/models/zoo.py`，在列表里加一行：

```python
for _name in [
    "resnet50",
    "your_new_model_name",   # ← 加这一行
    ...
]:
    register(_name)(_timm_factory(_name))
```

重新安装（`pip install -e .` 后无需重装，源码包已生效），然后配置里直接用新名字。

### 5.2 如果它是 torchvision 模型

`zoo.py` 里有 `_torchvision_factory` 帮你处理"换分类头"，加一行注册即可。

### 5.3 如果它是你自己写的网络（学习重点！）

以"从零写一个极简 CNN"为例。新建 `src/garbage_classifier/models/my_cnn.py`：

```python
"""一个从零手写的极简 CNN —— 教学示例。

结构：2 个卷积块 + 全局平均池化 + 全连接分类头。
这是 2012 年 AlexNet 级别的思路：卷积提取特征，全连接做分类。
"""
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        # 卷积块1：3→16 通道，提取边缘/纹理
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # 卷积块2：16→32 通道，提取更复杂的模式
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 下采样：224→112→56，逐步浓缩空间信息
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        # 全局平均池化：把 32×56×56 压成 32 维向量
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 分类头：32 维特征 → 6 类得分
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.gap(x).flatten(1)
        return self.head(x)
```

然后注册它（`zoo.py` 末尾）：

```python
from .registry import register

def _my_cnn_factory(num_classes, pretrained=False, **kwargs):
    from .my_cnn import MyCNN
    return MyCNN(num_classes=num_classes)

register("my_cnn")(_my_cnn_factory)
```

> 注意：`pretrained` 参数被接受但忽略 —— 手写网络没有预训练权重。从零训练
> 小网络在垃圾数据上精度有限，这是**正常的、值得亲身体会的事**：模型容量和
> 训练数据决定了天花板上限。

训练它：

```bash
garbage train --config configs/resnet50.yaml --set model.name my_cnn \
  --set model.pretrained false --set train.epochs 15 --set run_name mycnn-first
```

这个模型通常不如预训练 MobileNet，但具体差距取决于数据切分、随机初始化和训练配置。
先比较收敛速度与每类指标，再判断预训练带来了什么。

## 6. 理解 `garbage bench` 的输出

```bash
$ garbage bench
model                            params        MACs
convnext_tiny                    27.82M      4.48 GMac
mobilenetv3_small_100             1.52M     52.50 MMac
resnet50                         23.52M      4.12 GMac
```

- **params（参数量）**：模型"记住力"的规模，决定显存/内存占用。
- **MACs（乘加次数）**：一次前向的浮点运算量，决定推理速度。
- 学习点：`mobilenetv3` 用 1/15 的参数做到接近 resnet50 的精度 —— 这就是
  网络结构设计（深度可分离卷积）的价值；`convnext` 精度高但 FLOPs 也高。

## 7. 练习

```bash
# 练习 1：换三个模型跑 2 epoch 对比速度
for m in mobilenetv3_small_100 resnet50 convnext_tiny; do
  garbage train --config configs/resnet50.yaml --set model.name $m \
    --set train.epochs 2 --set data.num_workers 0 --set run_name speed-$m
done
# 看日志里每 epoch 耗时和 first-epoch 的 loss

# 练习 2：手写 my_cnn 并训练，对比精度
# （按 5.3 写完，训练 15 epoch，然后 evaluate）
garbage evaluate --checkpoint artifacts/mycnn-first/best.pt

# 练习 3：讲给朋友听 —— "backbone 和 head 各负责什么？为什么换模型只改配置？"
```

**下一步**：知道模型从哪来了，进入 [教程 4：训练策略选择](04-training.zh-CN.md) ——
优化器/学习率/batch size/增强/不平衡，每个决定背后是什么逻辑。
