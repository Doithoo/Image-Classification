# 教程 2：数据准备深潜

> 适用读者：想彻底搞懂"数据从哪来、去哪、变成什么"，以及**如何把自己的图片
> 数据集喂进这个项目**。
> 读完你会：能独立把任意"按类别分文件夹"的图片数据集准备好，并理解每一步
> 为什么存在。

## 1. 本项目需要的数据长什么样（最简单的情况）

```
你的数据目录/
├── cardboard/    ← 文件夹名 = 类别名（本项目是英文类别名）
│   ├── cardboard1.jpg
│   └── ...
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

**唯一要求**：每个类别一个文件夹，文件夹名就是类别名。图片格式支持
`.jpg/.jpeg/.png`。

本项目使用的公开数据集就是这个结构；下载后会先应用已记录的数据质量修正 —— 用
`python scripts/download_data.py` 下载后解压在 `data/raw/`。

---

## 2. 三步数据准备，每一步在做什么

```bash
# 步骤一：下载/准备原始数据（只需一次）
python scripts/download_data.py

# 步骤二：生成 manifest（切分 + 校验）
garbage prepare-data --set data.data_dir data/raw

# 步骤三：（不需要手动做）训练时读取 manifest + 变换
garbage train --config configs/resnet50.yaml
```

训练前先看一眼真实数据，避免把目录或标签问题带进训练：

```bash
python scripts/preview_dataset.py --data-dir data/raw
```

打开 `artifacts/dataset-preview.png`，确认每一行的类别和图片都符合预期；
`artifacts/class_counts.csv` 可直接用于比较类别数量。

### 步骤二内部发生了什么（`prepare-data` 的工作）

`garbage_classifier/data/manifest.py` 的 `build_manifest` 依次做四件事：

1. **扫描**：遍历 `data/raw/` 下每个类别文件夹，收集所有图片路径。
2. **校验**：逐张打开图片验证没损坏（`validate_image`）；另外按文件内容
   hash 检测重复图（`find_duplicates`，`--strict` 可让它拒绝继续）。
3. **分层切分**：对每个类别独立地、用固定随机种子（默认 666）打乱，再按
   8:1:1 分到 train/valid/test。**分层 = 每个类别在三个集合里的比例都一致**，
   避免某个类别全挤进训练集。
4. **产出**：三个 CSV manifest + 一个 `summary.txt`（数据摘要 + 校验和）。

### manifest 长什么样

```
$ head -3 data/manifests/train.csv
path,label
cardboard/cardboard233.jpg,0
cardboard/cardboard227.jpg,0
```

- `path` 是**相对路径**（相对数据根目录），这就是"可移植"的关键 —— 换机器、
  换目录都不影响。
- `label` 是类别索引，按类别名字母序：cardboard=0, glass=1, ..., trash=5。
  对应关系记录在 `data/manifests/source.yaml`（含数据根目录 + 类别表 + 校验和）。

### summary.txt 长什么样（数据摘要）

```
$ cat data/manifests/summary.txt
seed=666
split_ratios=[0.8, 0.1, 0.1]
classes=['cardboard', ...]
train=2019 valid=252 test=256
per_class_train={'0': 322, '1': 400, ...}
train_manifest_sha256=57ab9378...
```

**这就是数据审计记录**：任何人拿这份 summary 都能核对"这份切分是不是标准的那份"。

---

## 3. 理解切分（为什么 8:1:1，为什么固定种子）

| 集合 | 用途 | 能碰几次 |
|---|---|---|
| 训练集 (train) | 训练模型 | 每次迭代都碰 |
| 验证集 (valid) | 选模型、调超参、看训练曲线 | 随便看 |
| 测试集 (test) | 最终评估，衡量真实水平 | **只在最后碰一次** |

- 固定种子（666）→ 任何人重跑 `prepare-data` 得到**完全相同**的切分 →
  大家的实验结果才能互相比较。
- 测试集"只碰一次"是纪律：如果你拿测试集调参，报告的数字就不再代表真实水平。
  这是实验可信度的基本纪律：只有最后一次评估才使用测试集。

---

## 4. 换成其他数据时会变什么

项目读取 `数据目录/<类别名>/*.{jpg,jpeg,png}` 这一层。生成 manifest 后，类别表与
类别数来自 `source.yaml`，不需要手工修改 `model.num_classes`。

为了避免覆盖垃圾分类的 manifest，其他数据应使用单独的 manifest 目录。准备前先看
预览图，随后检查 `summary.txt` 中的类别、数量和切分。嵌套过深、坏图片、重复图片和
样本极少的类别都应该在长训练前处理。

完整命令和排查方法见[换成自己的图片数据](../guides/using-your-data.zh-CN.md)。

---

## 5. 数据变换（transforms）—— 训练前最后一步

图片进入模型前要变成"模型看得懂的张量"。`data/transforms.py` 里：

| 变换 | 训练集 | 验证/测试 | 作用 |
|---|---|---|---|
| `Resize(256)` | ✅ | ✅ | 把 512×384 缩放，短边=256（保持比例） |
| `RandomResizedCrop(224)` | ✅ | ❌ | 随机位置+随机缩放裁 224×224 → 数据多样性 |
| `CenterCrop(224)` | ❌ | ✅ | 固定裁中心 → 评测结果可复现 |
| `RandomHorizontalFlip` | ✅ | ❌ | 50% 概率水平翻转（物体翻转还是同类） |
| `RandAugment` | ✅(可选) | ❌ | 自动组合 2 个随机增强（对比度/旋转等） |
| `ToTensor()` | ✅ | ✅ | HWC uint8 [0,255] → CHW float32 [0,1] |
| `Normalize(mean,std)` | ✅ | ✅ | 归一化到近似标准正态，帮助收敛 |

**为什么训练和验证用不同变换**：训练要"多看花样"才能学会鲁棒特征；验证要
"每次看到的都一样"才能公平比较。两者混用会得出不可信的数字。

**归一化参数从哪来**：`mean=[0.673, 0.639, 0.604]` 等是该数据集像素的统计值
它存在 checkpoint 里（自包含），
所以训练/推理永远一致 —— 你不需要手动记它。

---

## 6. 可以快速确认的两件事

```bash
# 同一数据与种子是否得到相同切分？
cp data/manifests/train.csv /tmp/before.csv
garbage prepare-data --set data.data_dir data/raw
diff data/manifests/train.csv /tmp/before.csv && echo "完全一致（种子生效）"

# 训练变换与验证变换的输出形状是否一致？
python - <<'EOF'
from PIL import Image
from garbage_classifier.config import load_config
from garbage_classifier.data.transforms import build_train_transform, build_eval_transform
cfg = load_config()
train_t = build_train_transform(cfg.data)
eval_t = build_eval_transform(cfg.data)
img = Image.open("data/raw/cardboard/cardboard1.jpg")
print("原图:", img.size)
print("训练变换后:", tuple(train_t(img).shape), " 验证变换后:", tuple(eval_t(img).shape))
EOF
```

**下一步**：数据就绪，进入 [教程 3：模型构建与定制](03-model.zh-CN.md) ——
模型从哪来、怎么换、怎么从零写一个。
