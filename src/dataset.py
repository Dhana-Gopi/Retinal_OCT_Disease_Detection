"""PyTorch Dataset / DataLoader helpers and on-device data augmentation."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src import config
from src.preprocessing import load_cache, patient_split, to_model_tensor


class OCTArrayDataset(Dataset):
    """Serves preprocessed 224x224 uint8 scans from the memory-mapped cache.

    Returns raw uint8 tensors; normalisation (and augmentation for training)
    is applied batch-wise on the GPU in :func:`prepare_batch`, which is much
    faster on Apple Silicon than per-image CPU transforms.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.images = images
        self.labels = np.asarray(labels)
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        row = self.indices[item]
        image = torch.from_numpy(np.array(self.images[row]))  # copy out of memmap
        return image, int(self.labels[row])


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def augment_batch(batch01: torch.Tensor) -> torch.Tensor:
    """Random per-image augmentation of a [B, H, W] float batch in [0, 1].

    * horizontal flip (left/right eye mirror) with p = 0.5
    * small affine: rotation +-10 deg, scale 0.9-1.1, translation +-5 %
    * brightness +-15 % and contrast +-15 % (scanner gain variation)
    """
    b = batch01.shape[0]
    dev = batch01.device
    x = batch01.unsqueeze(1)
    angle = (torch.rand(b, device=dev) * 2 - 1) * math.radians(10)
    scale = 1 + (torch.rand(b, device=dev) * 2 - 1) * 0.10
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    tx, ty = [(torch.rand(b, device=dev) * 2 - 1) * 0.10 for _ in range(2)]
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1),
                         torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    brightness = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.15
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.15
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * brightness
    return x.clamp(0, 1).squeeze(1)


def prepare_batch(images_uint8: torch.Tensor, device: torch.device,
                  augment: bool = False) -> torch.Tensor:
    """uint8 [B,H,W] -> (optionally augmented) normalised float [B,3,H,W] on device."""
    x = images_uint8.to(device, non_blocking=True)
    if augment:
        x = augment_batch(x.float().div(255.0)).mul(255.0)
    return to_model_tensor(x)


def make_loaders(batch_size: int = 64, samples_per_epoch: int | None = None,
                 val_limit: int | None = None, num_workers: int = 0,
                 seed: int = config.SEED):
    """Build train / val / test DataLoaders.

    * Train: class-balanced sampling (WeightedRandomSampler) because the
      training split is imbalanced (NORMAL 47 % vs DRUSEN 8 %).  If
      ``samples_per_epoch`` is given, each "epoch" draws that many scans.
    * Val: patient-level hold-out from the training split (optionally a fixed
      stratified subset of ``val_limit`` scans for faster model selection).
    * Test: the supplied 1,000-scan test split, untouched until final evaluation.
    """
    train_images, train_meta = load_cache("train")
    test_images, test_meta = load_cache("test")
    train_idx, val_idx = patient_split(train_meta)
    labels = train_meta["label"].to_numpy()

    if val_limit and val_limit < len(val_idx):
        rng = np.random.default_rng(seed)
        per_class = val_limit // config.NUM_CLASSES
        val_idx = np.concatenate([
            rng.choice(val_idx[labels[val_idx] == c],
                       min(per_class, (labels[val_idx] == c).sum()), replace=False)
            for c in range(config.NUM_CLASSES)])
        val_idx.sort()

    class_counts = np.bincount(labels[train_idx], minlength=config.NUM_CLASSES)
    sample_weights = 1.0 / class_counts[labels[train_idx]]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                    num_samples=samples_per_epoch or len(train_idx),
                                    replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(OCTArrayDataset(train_images, labels, train_idx),
                              sampler=sampler, **common)
    val_loader = DataLoader(OCTArrayDataset(train_images, labels, val_idx),
                            shuffle=False, **common)
    test_loader = DataLoader(OCTArrayDataset(test_images, test_meta["label"].to_numpy(),
                                             np.arange(len(test_meta))),
                             shuffle=False, **common)
    return train_loader, val_loader, test_loader
