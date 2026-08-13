# 教程 1：设备与环境（CPU / NVIDIA GPU / Apple 芯片 / Kaggle）

> 适用读者：第一次在这个项目上跑训练，想知道"我这台电脑行不行、怎么装、跑多快"。
> 读完你会：装好环境、跑通 `--dry-run`、知道自己在哪种设备上、合理预期训练速度。

## 1. 先判断你的设备类型

| 设备 | 检测方法 | 本项目会用到的加速器 | 训练速度参考* |
|---|---|---|---|
| 纯 CPU 电脑 | 没有独立显卡 | 无（纯 CPU 计算） | resnet50 约 5–8 分钟/epoch |
| NVIDIA 显卡 | 终端跑 `nvidia-smi` | CUDA | resnet50 约 20–60 秒/epoch |
| Apple 芯片（M1/M2/M3） | 电脑是 Mac，芯片带 M 开头 | MPS | resnet50 约 1–2 分钟/epoch |
| Kaggle Notebook | 浏览器打开 kaggle.com | T4 GPU（免费） | resnet50 约 20–60 秒/epoch |

\* 粗略估计，随机器和 batch size 变化，仅作预期参考。

**原则**：本项目代码对设备完全透明 —— 配置里写 `device: auto`（默认），
框架会自动优先 CUDA → MPS → CPU。你不需要改代码，只需要装对 PyTorch 版本。

---

## 2. 通用准备（所有设备都需要）

```bash
# 1. Python 3.10+（建议 3.11）
python --version

# 2. 克隆项目
git clone https://github.com/Doithoo/Image-Classification.git
cd Image-Classification
```

**关于虚拟环境**：强烈建议用虚拟环境，避免污染系统 Python。本项目推荐 `uv`
（快、简单）：

```bash
pip install uv          # 只需装一次 uv 本身
uv venv --python 3.11 .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

> 不用 uv 也行：`python -m venv .venv && source .venv/bin/activate`。

---

## 3. 纯 CPU 电脑（Windows / Mac / Linux 无独显）

```bash
# 装 CPU 版 PyTorch（重点：不要装 CUDA 版，白占几个 GB）
# Linux/macOS:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Windows 也可以直接 pip install torch torchvision（默认就是 CPU 版）

# 然后装本项目
pip install -e ".[dev]"
```

**验证**：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.x.x False   （False 是正常的，我们走 CPU）
garbage train --config configs/resnet50.yaml --dry-run
```

**新手须知**：
- CPU 训练不是"不能跑"，而是"慢"。mobilenet_v3_small（1.5M 参数）在 CPU 上
  15 epoch 约 15–30 分钟，完全可以做完整练习；resnet50 请降低预期（或减少 epoch）。
- `--dry-run` 是你最好的朋友：长训练前先花 1 分钟确认管线没问题。
- 建议的 CPU 练习配置：`configs/imbalance_none.yaml` + `mobilenetv3_small_100`，
  15 epoch，`--set data.num_workers 0`（CPU 上多进程加载有时反而更慢）。

---

## 4. NVIDIA GPU（CUDA）

### 4.1 先确认你的显卡与驱动

```bash
nvidia-smi
# 期望看到显卡型号 + CUDA 版本（如 "CUDA Version: 12.4"）
```

- 没有输出 → 显卡驱动没装，去 NVIDIA 官网装驱动。
- 记下 `CUDA Version` 那一行（驱动支持的 CUDA 版本），决定装哪个 PyTorch。

### 4.2 安装对应 CUDA 的 PyTorch

PyTorch 官方按 CUDA 版本提供不同 wheel。去 [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
选你的系统，复制对应命令。常见：

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8（老显卡/老驱动）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> 装错版本最常见的症状：`torch.cuda.is_available()` 返回 False 且不报错。
> 此时检查 `nvidia-smi` 的驱动 CUDA 版本，重装匹配的 PyTorch。

### 4.3 验证

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: True NVIDIA GeForce ...
garbage train --config configs/resnet50.yaml --dry-run
# 日志里会打印 device: cuda
```

**新手须知**：
- 显存（VRAM）决定 batch size：8GB 显存建议 `batch_size 32` 以内（resnet50，
  224 输入）；OOM（out of memory）就把 batch size 减半，同时把学习率也减半
  （见教程 4 的"batch size 与学习率的关系"）。
- AMP（混合精度）默认开启（`train.amp: true`），能省显存并加速，不要关。
- `num_workers` 建议 4–8（GPU 太快，需要多进程加载数据才喂得饱）。

---

## 5. Apple 芯片（M1 / M2 / M3 / M4，macOS）

Apple 芯片自带统一内存，PyTorch 通过 **MPS** 后端调用 GPU。

```bash
pip install torch torchvision        # macOS 默认就是 MPS 版
pip install -e ".[dev]"
```

**验证**：
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# 期望: True
garbage train --config configs/resnet50.yaml --dry-run
# 日志里会打印 device: mps
```

**新手须知**：
- 16GB 内存的机器，batch size 32 对 resnet50 没问题；更大模型（vit/swin）建议
  16。OOM 就把 batch size 减半。
- MPS 的 `pin_memory` 不生效（代码已自动只在 CUDA 时启用），不影响使用。
- 风扇会响、机器会热 —— 正常。连续长时间训练建议让机器休息（本项目的实验
  就遇到过热降频导致训练变慢 40 倍，见 `docs/experiments.zh-CN.md`）。

---

## 6. Kaggle Notebook（免费 GPU，推荐体验）

Kaggle 免费提供 T4 GPU（约 16GB 显存），每周约 30 小时配额。适合：没有 GPU
电脑、想体验真实 GPU 训练速度的人。

### 6.1 新建 Notebook

1. 登录 [kaggle.com](https://www.kaggle.com) → Code → New Notebook
2. 右上角 **Settings**（⚙️）→ Accelerator 选 **GPU T4 x2**（或 T4 x1）
3. Language 选 **Python**

### 6.2 把代码弄进 Notebook（两种方式）

**方式 A（推荐，用 git）**：
```python
!git clone https://github.com/Doithoo/Image-Classification.git
%cd Image-Classification
!pip install -q -e ".[dev]"
```

**方式 B（上传）**：GitHub 页面 Code → Download ZIP → 上传到 Notebook 的
"Add data" → 再解压。

### 6.3 准备数据（两种方式）

**方式 A（从本项目 Release 下载，推荐）**：
```python
!python scripts/download_data.py        # 自动校验 SHA-256
!garbage prepare-data --set data.data_dir data/raw
```

**方式 B（把数据集作为 Kaggle Dataset 挂载）**：在 Notebook 右侧 "Add data"
搜 "garbage classification"，挂载后数据集在 `/kaggle/input/...`，然后：
```python
!garbage prepare-data --set data.data_dir /kaggle/input/你的数据集目录
```

> 注意：Kaggle 挂载的数据集是只读的，不要试图往里面写文件。

### 6.4 训练

```python
!garbage train --config configs/resnet50.yaml --set run_name kaggle-resnet50
```

设备自动识别为 `cuda`，日志里会看到 `device: cuda`。

**新手须知**：
- Kaggle 会话最长 12 小时，超过会断。长时间训练请用 `--resume` 续训：
  ```python
  !garbage train --config configs/resnet50.yaml --set run_name kaggle-resnet50 \
    --resume artifacts/kaggle-resnet50/last.pt
  ```
- 结果（`artifacts/`）存在临时磁盘，会话结束会丢！想保存：把
  `artifacts/<run>/best.pt` 和 `metrics.csv` 下载到本地，或上传到自己的
  GitHub Release / 网盘。
- 配额用完后要等下周；省着用：练习用 mobilenet（快 10 倍），确认无误再跑 resnet50。

---

## 7. 三个设备的通用验证清单

无论哪种设备，跑任何训练前先过这四关：

```bash
# 1. 环境装对
python -c "import torch; print(torch.__version__)"

# 2. 数据齐了
ls data/raw/cardboard | head -3
ls data/manifests/train.csv

# 3. 管线通了（1 分钟）
garbage train --config configs/resnet50.yaml --dry-run

# 4. 真跑一个小训练（确认速度符合预期）
garbage train --config configs/resnet50.yaml --set train.epochs 2 \
  --set data.num_workers 0 --set run_name speed-check
# 看日志里每 epoch 耗时，对比上表，若慢 10 倍以上说明哪里不对
```

## 8. 常见问题速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `torch.cuda.is_available()` 为 False | CUDA 版 PyTorch 与驱动不匹配 | 按 4.2 重装匹配版本 |
| 训练时 OOM | batch size 太大 | batch size 减半，学习率同步减半 |
| `--dry-run` 报错 | 数据路径不对 | 检查 `data/raw` 是否存在、`prepare-data` 是否跑过 |
| Kaggle 会话中断 | 12 小时限制 | 用 `--resume` 续训 |
| 机器很热、训练变慢 | 热降频 | 暂停让机器冷却 |
| `pip install` 很慢 | 网络 | 用镜像源：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...` |

**下一步**：环境就绪后，进入 [教程 2：数据准备](02-data.zh-CN.md) —— 搞清
"图片 → manifest → 批次"每一步在发生什么，以及如何用自己的数据。
