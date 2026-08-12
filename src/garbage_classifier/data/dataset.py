"""Dataset and collate function for the manifest-based image classifier.

Learning note — how PyTorch's Dataset/DataLoader contract works:
  - A Dataset answers two questions: "how many samples?" (__len__) and
    "give me sample #i" (__getitem__).
  - The DataLoader repeatedly calls __getitem__ with random indices, groups the
    results into batches, and feeds them to the training loop.
  - The dataset returns ONE raw (image, label) pair; the transform is applied
    lazily here so the same manifest can be reused with different transforms
    (train vs validation).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .manifest import load_manifest, manifest_root


class ImageClassificationDataset(Dataset):
    """Image dataset backed by a CSV manifest (portable relative paths).

    If ``root_dir`` is None, the data root recorded in the manifest directory is
    used, making manifests self-describing.
    """

    def __init__(
        self, manifest_path: str | Path, root_dir: str | Path | None = None, transform=None, target_transform=None
    ) -> None:
        root = Path(root_dir) if root_dir is not None else manifest_root(Path(manifest_path).parent)
        self.samples = load_manifest(manifest_path, root)  # [(absolute_path, label)]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")  # JPEGs may be grayscale/CMYK; force RGB
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge a list of (image, label) pairs into a batched tensor.

    Without a custom collate, DataLoader would use torch.stack on the list
    directly; we make it explicit and convert labels to a 1-D tensor.
    """
    images, labels = zip(*batch, strict=True)
    return torch.stack(images, dim=0), torch.as_tensor(labels)
