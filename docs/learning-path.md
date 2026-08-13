# Image Classification Learning Path (0 → 1)

> A guided curriculum for **beginners** that uses this repository as a scaffold
> to learn how to train image-classification models from scratch. Budget:
> **8–12 hours** including hands-on exercises. Do the stages in order.
>
> **中文版：[learning-path.zh-CN.md](learning-path.zh-CN.md)** ·
> Stuck? Hands-on tutorials (中文): [tutorials](tutorial/README.zh-CN.md)

## Prerequisites (optional but recommended)

- Basic Python: functions, classes, loops, `import`
- Basic PyTorch intuition: tensors, `nn.Module`, what `optimizer.step()` does
  (a 1-hour pass over the [PyTorch 60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) is enough)
- Basic ML concepts: train/validation/test split, overfitting, accuracy

**Not required**: reading papers, deriving math, building CNNs by hand.

---

## Stage 0: Environment (≈15 min)

**Goal**: get the project running and see your first training log.

```bash
cd Image-Classification
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
python scripts/download_data.py                    # downloads + verifies SHA-256
garbage prepare-data --set data.data_dir data/raw  # builds manifests
garbage train --config configs/resnet50.yaml --dry-run  # 1-batch sanity check
```

**Observe**: `data/manifests/` now has `train.csv / valid.csv / test.csv` and
`summary.txt`; the dry-run prints `dry-run OK: input=(32,3,224,224) output=(32,6) ...`
meaning data loads, the model builds, and forward+backward work.

**Read**: the "Quick start" and "Project structure" sections of `README.md`.

---

## Stage 1: The problem & the data (≈45 min)

**Goal**: understand the task — 6 garbage classes, and why plain accuracy lies.

**Read**: [`docs/how-it-works.md`](how-it-works.md) §1–2; the header comment of
`src/garbage_classifier/data/manifest.py`; `data/manifests/summary.txt`.

**Key ideas**:
1. **Class imbalance**: inspect the audited counts in
   `data/manifests/summary.txt`; use `balanced_accuracy` / `macro_f1` so a
   majority-class guesser cannot look useful.
2. **Train/valid/test discipline**: validation picks the model, the test set is
   touched only once at the end. Never tune on the test set.
3. **Why seed the split**: `seed=666` means everyone running `prepare-data` gets
   the identical split — the first step toward reproducible experiments.

**Check yourself**: use the generated summary to calculate the accuracy of a
model that always predicts the largest class. Its balanced accuracy is 1/6.

---

## Stage 2: The data pipeline (≈1 h)

**Goal**: trace the full journey of an image → tensor → batch.

**Read**: `src/garbage_classifier/data/dataset.py`, `transforms.py`,
`manifest.py` (`build_manifest` / `load_manifest`).

**Try**:
```bash
garbage prepare-data --set data.data_dir data/raw --strict
python -c "
from garbage_classifier.data import ImageClassificationDataset
from garbage_classifier.data.transforms import build_train_transform
from garbage_classifier.config import load_config
ds = ImageClassificationDataset('data/manifests/train.csv',
      transform=build_train_transform(load_config().data))
img, label = ds[0]
print(img.shape, label)   # torch.Size([3, 224, 224]) 0
"
```

After `download_data.py` applies the audited patch, strict mode is expected to
succeed. If you point it at the unpatched upstream v1 tree, its duplicate or
annotation-conflict diagnostic is expected and tells you which files to audit.

**Key ideas**:
1. **CSV manifests instead of folder walks**: relative paths + fixed split =
   portable and reproducible. (The old version used Windows backslash paths and
   crashed on macOS — the original motivation for this refactor.)
2. **Train vs validation transforms differ**: training needs diversity
   (`RandomResizedCrop`, flips, augmentation); validation needs determinism
   (`CenterCrop`).
3. **Normalization**: `Normalize(mean, std)` shifts pixels from [0,1] to roughly
   standard normal, which helps convergence. Train and inference must use the
   same values — our checkpoint carries them (Stage 6).

---

## Stage 3: Your first model (≈45 min)

**Goal**: understand the model registry and what a "backbone" is.

**Read**: `src/garbage_classifier/models/registry.py`, `zoo.py`, and the
`ModelConfig` in `src/garbage_classifier/config.py`.

**Try**:
```bash
garbage bench          # model sizes: params + FLOPs
garbage train --config configs/resnet50.yaml --set train.epochs 1 --set device cpu
```

**Key ideas**:
1. **Backbone + head**: a pretrained ResNet50 extracts features; we replace its
   1000-class head with our 6-class head (`num_classes=6`).
2. **Pretrained vs from-scratch**: `pretrained: true` initializes from ImageNet
   weights — on small datasets this usually crushes from-scratch (measured +27
   balanced-acc points in `docs/experiments.md` experiment 2).
3. **Why a registry**: `model.name` in YAML switches models without touching
   code. Config-driven beats edit-and-pray.

---

## Stage 4: The training loop (≈2 h — the core stage)

**Goal**: truly understand what happens inside one training iteration.

**Read (line by line, with the code open)**:
`src/garbage_classifier/training/trainer.py` (`_run_epoch` and `fit`),
`training/mixup.py` header, `training/ema.py` header, `training/weights.py`.

**Try (three comparable runs, ≈6 min each)**:
```bash
# A. baseline
garbage train --config configs/imbalance_none.yaml --set model.name mobilenetv3_small_100 \
  --set train.epochs 15 --set train.mixup_alpha 0 --set data.num_workers 0 --set run_name learn-a
# B. MixUp on
garbage train --config configs/imbalance_none.yaml --set model.name mobilenetv3_small_100 \
  --set train.epochs 15 --set train.mixup_alpha 0.2 --set data.num_workers 0 --set run_name learn-b

python scripts/plot_metrics.py artifacts/learn-a/metrics.csv
python scripts/plot_metrics.py artifacts/learn-b/metrics.csv
```

**Mental model of the loop**:
```
for epoch:
  for batch:
    outputs = model(images)              # forward: predict
    loss = loss_fn(outputs, labels)      # compare with truth
    loss.backward()                      # backward: gradients
    optimizer.step()                     # update weights along gradients
  scheduler.step()                       # anneal learning rate
  evaluate on validation → log metrics → update best.pt
  early stop if no improvement for N epochs
```

**Key ideas**:
1. **Warmup + cosine**: LR climbs from 1% to the target, then anneals to ~0.
   Early on the weights are far from optimal — a huge LR overshoots; late, a
   small LR lets them settle.
2. **MixUp**: blend two images (and their labels) with λ. The model must learn
   features, not memorize — a classic regularizer. **Caveat**: training loss
   looks higher with MixUp while validation improves. Don't be fooled by
   training loss.
3. **EMA**: a shadow of exponentially averaged weights; validate/save with the
   shadow, train with fast weights. Smoother, generalizes better. **But**: the
   decay time constant is 1/(1−decay) steps — `0.999` never catches up in a
   15-epoch run (experiment 4's lesson).
4. **Self-contained checkpoints**: `best.pt` holds config, class map, optimizer
   state — which is why resume (`--resume`) and reproducibility work.

---

## Stage 5: Evaluation (≈1 h)

**Goal**: read a report and explain every number.

**Read**: `src/garbage_classifier/evaluation/metrics.py`;
`docs/experiments.md` experiment 1 for a full example.

**Try**:
```bash
garbage evaluate --checkpoint artifacts/learn-b/best.pt --plot
```

**Key ideas**:
1. **Accuracy's blind spot**: 60 of 256 test images are paper — guessing paper
   yields 23.4%.
2. **Balanced accuracy**: the mean of per-class recall — every class counts
   equally.
3. **Precision / recall / F1** (using `trash` as the example):
   - precision = of the things predicted trash, how many are actually trash
   - recall = of the actual trash, how many were found
   - F1 = harmonic mean of the two
   - Rebalancing trades precision↔recall (experiment 3's lesson!).
4. **Confusion matrix**: rows = true, columns = predicted. The diagonal should
   be bright; *where* a class gets confused is where error analysis starts.

---

## Stage 6: Inference & explanation (≈45 min)

**Goal**: understand "inference only needs the checkpoint" and Grad-CAM.

**Read**: `src/garbage_classifier/inference/predictor.py`,
`inference/gradcam.py` header.

**Try**:
```bash
garbage predict --checkpoint artifacts/learn-b/best.pt --image data/raw/paper/paper1.jpg --top-k 3
garbage predict --checkpoint artifacts/learn-b/best.pt --image data/raw/paper/paper1.jpg --top-k 3 --tta
garbage explain --checkpoint artifacts/learn-b/best.pt --image data/raw/paper/paper1.jpg --output gradcam.png
```

**Key ideas**:
1. **Self-contained checkpoints**: `predict` needs no class names, normalization
   stats or model name — all restored from `best.pt`. No train/inference drift.
2. **TTA (test-time augmentation)**: flip the image, run once more, average the
   two softmaxes. Free accuracy (experiment 5: 0.906 → 0.922).
3. **Grad-CAM**: weight the last conv feature maps by the class-score gradient to
   see "where the model looked". When it errs, check whether it looked at the
   wrong region.

---

## Stage 7: Experiments & reproducibility (≈1 h)

**Goal**: turn "running experiments" into a discipline.

**Read**: `docs/experiments.md` (especially every "learning point"),
`src/garbage_classifier/training/checkpoint.py`.

**Try (your first ablation: learning rate)**:
```bash
garbage train --config configs/resnet50.yaml --set train.lr 1e-3 --set train.epochs 10 \
  --set data.num_workers 0 --set run_name myexp-lr1e3
garbage train --config configs/resnet50.yaml --set train.lr 1e-4 --set train.epochs 10 \
  --set data.num_workers 0 --set run_name myexp-lr1e4
python scripts/plot_metrics.py artifacts/myexp-lr1e3/metrics.csv
python scripts/plot_metrics.py artifacts/myexp-lr1e4/metrics.csv
```

**Key ideas (professional habits)**:
1. **One experiment = one artifact dir**: `config.yaml` (full config) +
   `metrics.csv` + `best.pt`. **config.yaml is the source of truth** — this very
   project once mislabeled pretrained fine-tuning as "from scratch"; only the
   config file exposed the mistake.
2. **Change one variable at a time**, or the comparison is meaningless.
3. **Record first, conclude second**: intuition ("rebalancing should help") gets
   contradicted by data all the time (experiment 3).

---

## Capstone project (optional, ≈2–3 h)

Use everything you learned in one closed loop:

1. Train a new model (`convnext_tiny`) for 20+ epochs
2. Compare accuracy/speed against resnet50
3. Use Grad-CAM to analyze its two worst classes
4. Export ONNX and build an interactive demo with `garbage demo`
5. Write it up as `docs/my-experiment.md`

---

## Map of docs & code

```
start → README.md (quick start)
  ↓
stage 0-2 → docs/how-it-works.md + data/ modules
  ↓
stage 3-4 → models/registry.py + training/{trainer,mixup,ema}.py
  ↓
stage 5-6 → evaluation/metrics.py + inference/{predictor,gradcam}.py
  ↓
stage 7 → docs/experiments.md + training/checkpoint.py
  ↓
capstone → combine the `garbage` commands freely
```

**Remember**: read the header comment of any module first (every learning module
explains *why*); `--dry-run` before long runs; check `config.yaml` before
concluding anything.
