"""Image preprocessing, dataset caching and patient-level train/val split.

The expensive JPEG decode + resize step is done once and stored as uint8
arrays on disk (``data/cache``).  Training then reads the arrays through a
memory map, which is far faster than decoding 108k JPEGs every epoch.
"""
from __future__ import annotations

import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src import config


# --------------------------------------------------------------------------
# Single-image preprocessing (used by the cache builder AND the Streamlit app)
# --------------------------------------------------------------------------
def preprocess_pil(image: Image.Image, size: int = config.IMAGE_SIZE,
                   mode: str = config.RESIZE_MODE) -> np.ndarray:
    """Grayscale -> resize to ``size`` x ``size``. Returns uint8 HxW.

    mode="stretch" (used for all final models): direct resize, no padding bars.
    mode="pad": keep aspect ratio and pad with black (earlier experiment only).
    """
    gray = ImageOps.exif_transpose(image).convert("L")
    if mode == "pad":
        resized = ImageOps.pad(gray, (size, size), method=Image.BILINEAR,
                               color=config.PAD_FILL)
    elif mode == "stretch":
        resized = gray.resize((size, size), Image.BILINEAR)
    else:
        raise ValueError(f"Unknown resize mode: {mode}")
    return np.asarray(resized, dtype=np.uint8)


def to_model_tensor(gray_uint8: np.ndarray | torch.Tensor) -> torch.Tensor:
    """uint8 array(s) [..., H, W] -> normalised float tensor [..., 3, H, W]."""
    tensor = torch.as_tensor(gray_uint8)
    tensor = tensor.float().div(255.0).unsqueeze(-3).repeat_interleave(3, dim=-3)
    mean = torch.tensor(config.IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Inverse of :func:`to_model_tensor` for display (returns HxW float 0-1)."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    return img[0].clamp(0, 1).numpy()


def load_image_bytes(data: bytes) -> Image.Image:
    """Open image bytes, raising ValueError for corrupted / unsupported files."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
        raise ValueError(f"Invalid or corrupted image: {err}") from err
    return image


# --------------------------------------------------------------------------
# Cache builder
# --------------------------------------------------------------------------
def _hf_parquet_files(split: str) -> list[Path]:
    """Return the locally cached parquet shards for a split (no download)."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(config.HF_DATASET_ID, repo_type="dataset",
                                  allow_patterns=[f"data/{split}-*.parquet"],
                                  local_files_only=True))
    return sorted((root / "data").glob(f"{split}-*.parquet"))


def _process_shard(parquet_path: str, mode: str = config.RESIZE_MODE) -> tuple[np.ndarray, list[dict]]:
    """Decode + preprocess every image of one parquet shard."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["image", "label"])
    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()
    arrays, rows = [], []
    for item, label in zip(images, labels):
        file_name = item["path"]
        try:
            pil = load_image_bytes(item["bytes"])
            orig_w, orig_h = pil.size
            arrays.append(preprocess_pil(pil, mode=mode))
        except ValueError:
            # Corrupted image: skip it and record nothing (counted below).
            rows.append({"file_name": file_name, "label": label, "corrupt": True})
            continue
        rows.append({"file_name": file_name, "label": label, "corrupt": False,
                     "patient_id": file_name.split("-")[1],
                     "orig_width": orig_w, "orig_height": orig_h})
    return np.stack(arrays), rows


def build_cache(split: str, workers: int = 8, cache_dir: Path = config.CACHE_DIR,
                mode: str = config.RESIZE_MODE) -> None:
    """Preprocess an entire split into ``{cache_dir}/{split}_images.npy`` + metadata CSV."""
    from functools import partial

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    images_path = cache_dir / f"{split}_images.npy"
    meta_path = cache_dir / f"{split}_meta.csv"
    if images_path.exists() and meta_path.exists():
        print(f"[cache] {split} already cached -> {images_path}")
        return
    import pyarrow.parquet as pq

    shards = [str(p) for p in _hf_parquet_files(split)]
    total = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    # Write straight to a disk-backed array so 5+ GB never sits in RAM twice.
    images = np.lib.format.open_memmap(images_path.with_suffix(".tmp.npy"), mode="w+",
                                       dtype=np.uint8,
                                       shape=(total, config.IMAGE_SIZE, config.IMAGE_SIZE))
    all_rows, cursor = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for arrays, rows in pool.map(partial(_process_shard, mode=mode), shards):
            images[cursor:cursor + len(arrays)] = arrays
            cursor += len(arrays)
            all_rows.extend(rows)
            print(f"[cache] {split}: {cursor:,}/{total:,} images done", flush=True)
    images.flush()
    del images
    meta = pd.DataFrame(all_rows)
    corrupt = meta[meta["corrupt"]]
    meta = meta[~meta["corrupt"]].drop(columns="corrupt").reset_index(drop=True)
    assert len(meta) == cursor
    # Rows after `cursor` (only if images were corrupt) stay unused zeros.
    images_path.with_suffix(".tmp.npy").rename(images_path)
    meta.to_csv(meta_path, index=False)
    with open(cache_dir / f"{split}_corrupt.json", "w") as fh:
        json.dump(corrupt["file_name"].tolist(), fh)
    print(f"[cache] saved {len(meta):,} images ({len(corrupt)} corrupt skipped)")


def load_cache(split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Memory-map the cached images (no RAM copy) and load the metadata."""
    images = np.load(config.CACHE_DIR / f"{split}_images.npy", mmap_mode="r")
    meta = pd.read_csv(config.CACHE_DIR / f"{split}_meta.csv", dtype={"patient_id": str})
    return images, meta


# --------------------------------------------------------------------------
# Patient-level split (prevents leakage of one patient's scans across sets)
# --------------------------------------------------------------------------
def patient_split(meta: pd.DataFrame, val_fraction: float = config.VAL_PATIENT_FRACTION,
                  seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) with every patient entirely in one side."""
    split_file = config.CACHE_DIR / "patient_split.json"
    if split_file.exists():
        saved = json.loads(split_file.read_text())
        return np.array(saved["train_idx"]), np.array(saved["val_idx"])
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = round(1 / val_fraction)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(meta, meta["label"], groups=meta["patient_id"]))
    assert not set(meta.patient_id.iloc[train_idx]) & set(meta.patient_id.iloc[val_idx])
    split_file.write_text(json.dumps({"train_idx": train_idx.tolist(),
                                      "val_idx": val_idx.tolist()}))
    return train_idx, val_idx


if __name__ == "__main__":
    import sys

    # optional: python -m src.preprocessing <cache_dir> <mode>
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else config.CACHE_DIR
    resize_mode = sys.argv[2] if len(sys.argv) > 2 else config.RESIZE_MODE
    build_cache("test", cache_dir=target, mode=resize_mode)
    build_cache("train", cache_dir=target, mode=resize_mode)
    if target != config.CACHE_DIR:
        raise SystemExit(0)
    _, train_meta = load_cache("train")
    tr, va = patient_split(train_meta)
    print(f"train={len(tr):,} val={len(va):,} "
          f"val patients={train_meta.patient_id.iloc[va].nunique():,}")
