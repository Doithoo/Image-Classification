# Baseline 冻结报告

> 目的：在重构前锁定当前项目的真实状态 —— 环境、可运行性、数据集、模型、历史指标与已知缺陷，作为后续所有改动的对照基准。
> 冻结时间：重构启动时（git revision: `bff744a`）

## 1. 环境记录

### 1.1 原始开发环境（历史推断，非本机实测）

- 代码生成年代：Python 3.9（`Code/` 下遗留 `cpython-39.pyc`）
- 训练/推理曾在 **Windows 与 Kaggle** 设备上进行：
  - `train.txt` 等清单为 Windows 反斜杠路径（`..\train\...`）
  - `Code/data_prepare/02_generate_txt.py` 中写死 Kaggle 绝对路径（`/kaggle/working/Image-Classification`）
- 依赖为散装安装（无 requirements / lock 文件）：torch、torchvision、numpy、PIL、cv2、matplotlib、tqdm、prettytable、ptflops、torchinfo、lion-pytorch

### 1.2 基线验证环境（本次实测）

| 项 | 值 |
|---|---|
| 设备 | Apple Silicon (MPS 可用，无 CUDA) |
| Python | 3.11.7 (pyenv, 项目 venv `.venv/`) |
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| timm | 1.0.28 |
| 其余 | numpy / pillow / matplotlib / tqdm / prettytable / ptflops / torchinfo / opencv-python / lion-pytorch / pyyaml 已装 |

## 2. 可运行性结论（关键）

**当前代码在本机（macOS/Linux）不可直接运行，原因是指纹级路径问题：**

| 检查项 | 结果 |
|---|---|
| `train.txt` 反斜杠路径可打开 | 0 / 2019 |
| 转正斜杠后从 `Code/` 运行可打开 | 2019 / 2019 |
| `MyDataset` 解析 txt | 成功（仅解析不读图） |
| `MyDataset.__getitem__` | **FileNotFoundError**：`'..\\train\\cardboard\\cardboard1.jpg'` |
| `train.py` / `predict.py` | 无法在本机执行训练/推理（撞上同一问题） |

结论：`MyDataset` 依赖平台相关的路径分隔符，代码仅在 Windows（或按 Kaggle 绝对路径）可用。这是重构中必须解决的首要可移植性问题。

## 3. 数据集现状

- 原始集 `Garbage_classification/` 与切分副本 `train/valid/test/` 同时入库，共 **5054 张**（2527 张唯一图片 × 2 份冗余副本）。
- 坏图检查：**0 张坏图**；抽样尺寸全部 **512×384**。
- 切分比例 8:1:1，且按类别分层（`01_split_dataset.py`，随机种子 666）。

### 类别分布

| 类别 | train | valid | test | train 占比 |
|---|---:|---:|---:|---:|
| cardboard | 322 | 40 | 41 | 15.9% |
| glass | 400 | 50 | 51 | 19.8% |
| metal | 328 | 41 | 41 | 16.2% |
| paper | 475 | 59 | 60 | 23.5% |
| plastic | 385 | 48 | 49 | 19.1% |
| trash | **109** | 14 | 14 | **5.4%** |
| **合计** | 2019 | 252 | 256 | 100% |

**类别不平衡确认**：trash 仅占训练集 5.4%，paper 是其 4.4 倍 —— 仅用 Accuracy 评估会掩盖 trash 类表现，评测必须包含 balanced accuracy / per-class metrics / macro-F1。

### txt 清单与目录结构一致性

- 标签与目录类别映射一致（2019/2019 无错位），`02_generate_txt.py` 的类别排序逻辑正确。

## 4. 模型现状

16 个模型入口全部可在 torch 2.13 下构建并完成 224×224 前向传播（随机权重，CPU）：

| 模型 | 参数量 | 前向 | 说明 |
|---|---:|---|---|
| alexnet | 14.6M | ✅ | `AlexNet(num_classes)` |
| vgg11 | 128.8M | ✅ | 须用关键字 `vgg(model_name=..., num_classes=...)` |
| resnet18 / resnet50 | 11.2M / 23.5M | ✅ | |
| googlenet | 6.0M | ✅ | `GoogLeNet(num_classes=, aux_logits=False)` |
| densenet121 | 7.0M | ✅ | `densenet121(num_classes=...)` |
| mobilenet_v2 | 2.2M | ✅ | 工厂名为 `MobileNetV2` 类 |
| mobilenet_v3_small / large | 1.5M / — | ✅ | 仅测 small |
| efficientnet_b0 / efficientnetv2_s | 4.0M / 20.2M | ✅ | |
| regnet | 2.3M | ✅ | `regnet(num_classes=...)` |
| shufflenet_v2_x0_5 | 0.3M | ✅ | |
| convnext_tiny | 27.8M | ✅ | |
| vit_base_patch16_224 | 85.8M | ✅ | |
| swin_tiny | 27.5M | ✅ | 当前 train/predict 默认选中 |

> 注：以上均为**随机初始化**验证（未训练、无 pretrain 权重）。模型无统一注册接口、无依赖版本约束。

## 5. 历史指标（来自 README，不可追溯）

下表来自旧 README，在**其他设备**上训练所得；仓库内无对应配置/种子/权重，**不可复现**，仅作参考：

| 模型 | Acc | Params | FLOPs |
|---|---:|---:|---:|
| alexnet | 0.816 | 14.59M | 310.07M |
| vgg11 | 0.855 | 128.79M | 7.63G |
| vgg13 | 0.859 | 128.98M | 11.34G |
| vgg16 | 0.832 | 134.29M | 15.5G |
| vgg19 | 0.824 | 139.59M | 19.66G |
| resnet18 | 0.855 | 11.18M | 1.82G |
| resnet34 | 0.867 | 21.29M | 3.68G |
| resnet50 | 0.871 | 23.52M | 4.12G |
| resnet101 | 0.832 | 42.51M | 7.85G |

## 6. 已确认缺陷清单（重构时的对照用例）

1. **路径不可移植**：txt 反斜杠 + Kaggle 绝对路径（见 §2）。
2. **归一化参数不一致**：`train.py` 用 `[0.673, 0.639, 0.604]`（四舍五入），`predict.py` 用 `[0.67254436, 0.639278, 0.6043134]`（高精度），两者不匹配。
3. **混淆矩阵语义反置**：`utils/confusion_matrix.py` 中 `matrix[p, t]`（行=预测、列=真实），与常规 `matrix[true, pred]` 相反，plot 坐标与矩阵索引方向易混淆。
4. **无 checkpoint 元数据**：只存 `state_dict`，无类别映射/预处理/配置，推理需手工重复全部配置。
5. **无断点续训/早停/AMP/EMA**；`epochs=300` 无 resume 实际不可用。
6. **随机种子不完整**：仅 `torch.manual_seed(1)`，未覆盖 numpy/random/cuda/mps。
7. **数据三份冗余**（92MB 入库）+ `__pycache__` 提交进 git。

## 7. 后续对照基准（Refactoring Contract）

重构完成后，以下断言必须成立：

- [ ] 全新环境可 `pip install -e .` + 一条 CLI 命令完成 CPU smoke 训练
- [ ] 数据集可移植（任意平台、任意工作目录可运行）
- [ ] 训练/推理共用同一份归一化参数（来自 config 或 checkpoint）
- [ ] 任何完整实验可凭保存的配置与数据版本复现
- [ ] 测试集从未参与调参；所有模型比较可追溯（配置+种子+权重归档）
- [ ] 推理不需要手工指定类别名、均值方差、模型名
- [ ] 评测输出包含 balanced accuracy / macro-F1 / per-class P-R / 混淆矩阵
