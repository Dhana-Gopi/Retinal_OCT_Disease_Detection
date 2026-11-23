"""Dataset sanity checks and exploratory figures for the Retinal OCT project.

Run from the project root (CPU only, does not touch the GPU)::

    python -m src.data_checks

Verifies
    1. class order: HF ``ClassLabel`` names == ``config.CLASS_NAMES``, and the
       file-name prefix of every row in the cached metadata matches its label;
       the cached metadata order matches the parquet row order.
    2. cache correctness: a random sample of train / test images is decoded
       from the original parquet bytes, re-preprocessed with
       ``preprocess_pil`` and compared byte-for-byte with the cached arrays.
    3. patient split: no patient in both train and val, no patient shared
       between the train pool (train + val) and the test split; counts per
       class / split and patient counts.
    4. original image-size distribution (train and test, per class) and
       flags sizes that identify a class (shortcut risk) or differ by split.
    5. class counts of the fixed 4,000-scan stratified validation subset used
       for model selection (``make_loaders(val_limit=4000)``).

Writes ``outputs/metrics/data_report.json`` and ``outputs/figures/data_*.png``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no GUI backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from src import config  # noqa: E402
from src.preprocessing import (  # noqa: E402
    _hf_parquet_files,
    load_cache,
    load_image_bytes,
    patient_split,
    preprocess_pil,
)

# Fixed colour per class (validated categorical palette, slots 1-4), so a
# class has the same colour in every figure.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
DPI = 120
N_CHECK_TRAIN = 50
N_CHECK_TEST = 20
VAL_LIMIT = 4000


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _style_axes(ax: plt.Axes) -> None:
    """Recessive axes: no top/right spines, light horizontal grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    """Count labels per class name (in the fixed class order)."""
    counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    return {name: int(c) for name, c in zip(config.CLASS_NAMES, counts)}


def _split_summary(meta, idx: np.ndarray) -> dict:
    """Scan counts per class, total scans and number of patients for a split."""
    sub = meta.iloc[idx]
    return {"scans": int(len(sub)),
            "patients": int(sub["patient_id"].nunique()),
            "per_class": _class_counts(sub["label"].to_numpy()),
            "patients_per_class": {
                config.CLASS_NAMES[c]: int(sub.loc[sub.label == c, "patient_id"].nunique())
                for c in range(config.NUM_CLASSES)}}


def _hf_class_names(shard: Path) -> list[str]:
    """Read the ClassLabel names stored in the parquet's HF schema metadata."""
    meta = pq.ParquetFile(shard).schema_arrow.metadata[b"huggingface"]
    return json.loads(meta)["info"]["features"]["label"]["names"]


def _parquet_paths_labels(shards: list[Path]) -> tuple[list[str], np.ndarray]:
    """File names and labels of every row, in shard order (no image bytes)."""
    paths, labels = [], []
    for shard in shards:
        table = pq.read_table(shard, columns=["image.path", "label"])
        paths.extend(table.column("path").to_pylist())
        labels.extend(table.column("label").to_pylist())
    return paths, np.asarray(labels)


def _fetch_rows(shards: list[Path], global_rows: np.ndarray) -> dict[int, dict]:
    """Fetch image bytes/path/label for given global row numbers.

    Only the parquet row groups that contain the requested rows are read.
    """
    # Build a (shard, row group) -> global row offset table.
    locations, offset = [], 0
    for shard in shards:
        meta = pq.ParquetFile(shard).metadata
        for rg in range(meta.num_row_groups):
            n = meta.row_group(rg).num_rows
            locations.append((shard, rg, offset, n))
            offset += n
    wanted = set(int(r) for r in global_rows)
    result: dict[int, dict] = {}
    for shard, rg, start, n in locations:
        hits = [r for r in wanted if start <= r < start + n]
        if not hits:
            continue
        table = pq.ParquetFile(shard).read_row_group(rg, columns=["image", "label"])
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        for r in hits:
            result[r] = {"bytes": images[r - start]["bytes"],
                         "path": images[r - start]["path"],
                         "label": labels[r - start]}
    return result


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_class_order(shards: dict[str, list[Path]], metas: dict) -> dict:
    """ClassLabel names, filename prefixes and meta-vs-parquet row order."""
    report = {}
    hf_names = _hf_class_names(shards["test"][0])
    report["hf_class_names"] = hf_names
    report["config_class_names"] = list(config.CLASS_NAMES)
    report["class_order_matches"] = hf_names == list(config.CLASS_NAMES)
    for split, meta in metas.items():
        prefix = meta["file_name"].str.split("-").str[0]
        expected = meta["label"].map(dict(enumerate(config.CLASS_NAMES)))
        mismatches = int((prefix != expected).sum())
        paths, labels = _parquet_paths_labels(shards[split])
        corrupt = json.loads((config.CACHE_DIR / f"{split}_corrupt.json").read_text())
        order_ok = (not corrupt and paths == meta["file_name"].tolist()
                    and np.array_equal(labels, meta["label"].to_numpy()))
        report[split] = {"rows_in_meta": int(len(meta)),
                         "rows_in_parquet": len(paths),
                         "corrupt_skipped": len(corrupt),
                         "filename_prefix_label_mismatches": mismatches,
                         "meta_order_equals_parquet_order": bool(order_ok),
                         "duplicate_file_names": int(meta["file_name"].duplicated().sum())}
    report["passed"] = bool(report["class_order_matches"] and all(
        report[s]["filename_prefix_label_mismatches"] == 0
        and report[s]["meta_order_equals_parquet_order"] for s in metas))
    return report


def check_cache(shards: dict[str, list[Path]], caches: dict, metas: dict,
                seed: int = config.SEED) -> dict:
    """Re-preprocess random originals and compare byte-for-byte with the cache."""
    rng = np.random.default_rng(seed)
    report = {}
    for split, n in (("train", N_CHECK_TRAIN), ("test", N_CHECK_TEST)):
        rows = np.sort(rng.choice(len(metas[split]), n, replace=False))
        fetched = _fetch_rows(shards[split], rows)
        equal, name_ok, failures = 0, 0, []
        for r in rows:
            item = fetched[int(r)]
            fresh = preprocess_pil(load_image_bytes(item["bytes"]))
            same = (fresh.shape == caches[split][r].shape
                    and np.array_equal(fresh, caches[split][r]))
            name_match = item["path"] == metas[split]["file_name"].iloc[r]
            equal += int(same)
            name_ok += int(name_match)
            if not (same and name_match):
                failures.append(item["path"])
        report[split] = {"checked": int(n), "byte_equal": equal,
                         "file_name_matches": name_ok, "failures": failures}
    report["passed"] = all(report[s]["byte_equal"] == report[s]["checked"]
                           and report[s]["file_name_matches"] == report[s]["checked"]
                           for s in ("train", "test"))
    return report


def check_patient_split(metas: dict, train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
    """Patient leakage checks plus per-class / per-split counts."""
    train_meta, test_meta = metas["train"], metas["test"]
    tr_p = set(train_meta.patient_id.iloc[train_idx])
    va_p = set(train_meta.patient_id.iloc[val_idx])
    pool_p = set(train_meta.patient_id)
    te_p = set(test_meta.patient_id)
    idx_disjoint = not set(train_idx.tolist()) & set(val_idx.tolist())
    covers_all = len(train_idx) + len(val_idx) == len(train_meta)
    report = {
        "train": _split_summary(train_meta, train_idx),
        "val": _split_summary(train_meta, val_idx),
        "test": _split_summary(test_meta, np.arange(len(test_meta))),
        "train_pool_patients": len(pool_p),
        "patients_train_and_val": len(tr_p & va_p),
        "patients_train_pool_and_test": len(pool_p & te_p),
        "train_val_indices_disjoint": idx_disjoint,
        "train_val_cover_all_rows": covers_all,
        "patients_with_multiple_classes_train_pool": int(
            (train_meta.groupby("patient_id")["label"].nunique() > 1).sum()),
    }
    report["passed"] = (report["patients_train_and_val"] == 0
                        and report["patients_train_pool_and_test"] == 0
                        and idx_disjoint and covers_all)
    return report


def image_size_distribution(meta) -> dict:
    """Counts of each original (width x height), overall and per class."""
    sizes = meta.orig_width.astype(str) + "x" + meta.orig_height.astype(str)
    counts = sizes.value_counts()
    by_class = {size: {config.CLASS_NAMES[c]: int(((sizes == size) & (meta.label == c)).sum())
                       for c in range(config.NUM_CLASSES)} for size in counts.index}
    return {"n_images": int(len(meta)),
            "distinct_sizes": int(len(counts)),
            "size_counts": {k: int(v) for k, v in counts.items()},
            "size_counts_by_class": by_class,
            "width": {"min": int(meta.orig_width.min()),
                      "max": int(meta.orig_width.max()),
                      "median": float(meta.orig_width.median())},
            "height": {"min": int(meta.orig_height.min()),
                       "max": int(meta.orig_height.max()),
                       "median": float(meta.orig_height.median())}}


def size_shortcut_risks(train_sizes: dict, test_sizes: dict) -> list[str]:
    """Flag image sizes that identify a class, or that differ between splits."""
    notes = []
    for split, info in (("train", train_sizes), ("test", test_sizes)):
        for size, per_class in info["size_counts_by_class"].items():
            total = sum(per_class.values())
            top = max(per_class, key=per_class.get)
            if total >= 100 and per_class[top] / total >= 0.95:
                notes.append(f"{split}: {size} ({total} scans) is {per_class[top] / total:.0%} "
                             f"{top} - image size/padding can act as a class shortcut")
    tr_top = max(train_sizes["size_counts"], key=train_sizes["size_counts"].get)
    te_top = max(test_sizes["size_counts"], key=test_sizes["size_counts"].get)
    if tr_top != te_top:
        tr_share = train_sizes["size_counts"].get(te_top, 0) / train_sizes["n_images"]
        notes.append(f"most common size differs: train {tr_top}, test {te_top} "
                     f"({te_top} is {tr_share:.1%} of train) - aspect-ratio/padding shift")
    return notes


def check_val_subset(train_meta) -> dict:
    """Class counts of the fixed stratified val subset used for model selection."""
    from src.dataset import make_loaders

    _, val_loader, _ = make_loaders(batch_size=64, val_limit=VAL_LIMIT)
    ds = val_loader.dataset
    labels = ds.labels[ds.indices]
    _, val_loader2, _ = make_loaders(batch_size=64, val_limit=VAL_LIMIT)
    reproducible = np.array_equal(ds.indices, val_loader2.dataset.indices)
    val_idx = patient_split(train_meta)[1]
    available = np.bincount(ds.labels[val_idx], minlength=config.NUM_CLASSES)
    expected = int(np.minimum(available, VAL_LIMIT // config.NUM_CLASSES).sum())
    return {"val_limit": VAL_LIMIT,
            "expected_scans": expected,
            "note": ("subset is capped at val_limit // 4 per class; classes with fewer "
                     "val scans are taken whole, so the subset can be < val_limit"),
            "scans": int(len(ds.indices)),
            "patients": int(train_meta.patient_id.iloc[ds.indices].nunique()),
            "per_class": _class_counts(labels),
            "reproducible_across_calls": bool(reproducible),
            "subset_within_val": bool(set(ds.indices.tolist()) <= set(val_idx.tolist())),
            "passed": bool(len(ds.indices) == expected and reproducible
                           and set(ds.indices.tolist()) <= set(val_idx.tolist()))}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_class_distribution(split_report: dict, val_subset: dict, path: Path) -> None:
    """Bar chart of scans per class for each split (one panel per split)."""
    panels = [("Train", split_report["train"]), ("Validation", split_report["val"]),
              (f"Val subset (val_limit={VAL_LIMIT:,})", val_subset),
              ("Test", split_report["test"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.2))
    for ax, (title, info) in zip(axes, panels):
        counts = [info["per_class"][c] for c in config.CLASS_NAMES]
        bars = ax.bar(config.CLASS_NAMES, counts, color=CLASS_COLORS, width=0.7,
                      edgecolor="white", linewidth=2)
        total = sum(counts)
        for bar, c in zip(bars, counts):
            ax.annotate(f"{c:,}\n{c / total:.0%}", (bar.get_x() + bar.get_width() / 2, c),
                        ha="center", va="bottom", fontsize=8, color=TEXT,
                        xytext=(0, 2), textcoords="offset points")
        patients = info.get("patients")
        ax.set_title(f"{title}\n{total:,} scans, {patients:,} patients",
                     fontsize=10, color=TEXT)
        ax.set_ylim(0, max(counts) * 1.22)
        _style_axes(ax)
    axes[0].set_ylabel("Number of scans", color=MUTED)
    fig.suptitle("Scans per class in each split (patient-level split)", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_samples(shards: list[Path], test_meta, path: Path, seed: int = config.SEED) -> None:
    """Original test scan vs the 224x224 preprocessed version, one per class."""
    rng = np.random.default_rng(seed)
    rows = [int(rng.choice(np.flatnonzero(test_meta.label.to_numpy() == c)))
            for c in range(config.NUM_CLASSES)]
    fetched = _fetch_rows(shards, np.array(rows))
    fig, axes = plt.subplots(2, config.NUM_CLASSES, figsize=(15, 6.5),
                             gridspec_kw={"height_ratios": [1, 1.25]})
    for c, r in enumerate(rows):
        pil = load_image_bytes(fetched[r]["bytes"])
        axes[0, c].imshow(np.asarray(pil.convert("L")), cmap="gray", vmin=0, vmax=255)
        axes[0, c].set_title(f"{config.CLASS_NAMES[c]} - original {pil.size[0]}x{pil.size[1]}"
                             f"\n{fetched[r]['path']}", fontsize=9, color=TEXT)
        axes[1, c].imshow(preprocess_pil(pil), cmap="gray", vmin=0, vmax=255)
        axes[1, c].set_title("preprocessed 224x224 (aspect kept, padded)",
                             fontsize=9, color=MUTED)
        for ax in axes[:, c]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Test-split OCT scans: original vs model input", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_augmentation(train_cache: np.ndarray, train_meta, path: Path,
                     seed: int = config.SEED) -> None:
    """One training scan and six random augmentations (augment_batch, CPU)."""
    from src.dataset import augment_batch

    row = int(np.flatnonzero(train_meta.label.to_numpy() == 0)[0])
    original = torch.from_numpy(np.array(train_cache[row])).float().div(255.0)
    torch.manual_seed(seed)
    augmented = augment_batch(original.unsqueeze(0).repeat(6, 1, 1)).numpy()
    fig, axes = plt.subplots(1, 7, figsize=(17, 3.1))
    axes[0].imshow(original.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original\n{train_meta.file_name.iloc[row]}", fontsize=9, color=TEXT)
    for i, img in enumerate(augmented, start=1):
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"Augmentation {i}", fontsize=9, color=MUTED)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Training augmentation: flip, rotate +-10 deg, scale +-10 %, "
                 "shift, brightness/contrast +-15 %", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_image_sizes(size_reports: dict, path: Path) -> None:
    """Stacked horizontal bars: original image sizes per split, split by class."""
    fig, axes = plt.subplots(1, len(size_reports), figsize=(14, 4.2))
    for ax, (split, info) in zip(axes, size_reports.items()):
        sizes = list(info["size_counts"])[::-1]  # largest at top
        total = info["n_images"]
        left = np.zeros(len(sizes))
        for c, name in enumerate(config.CLASS_NAMES):
            vals = np.array([info["size_counts_by_class"][s][name] for s in sizes])
            ax.barh(sizes, vals, left=left, color=CLASS_COLORS[c], height=0.65,
                    edgecolor="white", linewidth=1, label=name)
            left += vals
        for y, c in enumerate(left):
            ax.annotate(f"{int(c):,} ({c / total:.1%})", (c, y), va="center", fontsize=8,
                        color=TEXT, xytext=(3, 0), textcoords="offset points")
        ax.set_xlim(0, left.max() * 1.3)
        ax.set_xlabel("Number of scans", color=MUTED)
        ax.set_title(f"{split.capitalize()} split (n={total:,})", color=TEXT, fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("Original size (width x height)", color=MUTED)
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Original image sizes by class: 512x512 is DME-only in train; "
                 "test is mostly 1024x496", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def intensity_stats(train_cache: np.ndarray, train_meta, train_idx: np.ndarray,
                    per_class: int = 500, seed: int = config.SEED) -> dict:
    """Mean pixel intensity (preprocessed, padding excluded) of sampled train scans."""
    rng = np.random.default_rng(seed)
    labels = train_meta.label.to_numpy()
    out = {}
    for c, name in enumerate(config.CLASS_NAMES):
        pool = train_idx[labels[train_idx] == c]
        rows = np.sort(rng.choice(pool, min(per_class, len(pool)), replace=False))
        imgs = np.asarray(train_cache[rows], dtype=np.float32)
        # Exclude the black padding rows/cols so the aspect ratio does not bias it.
        content = imgs > 0
        means = (imgs * content).sum(axis=(1, 2)) / np.maximum(content.sum(axis=(1, 2)), 1)
        out[name] = means
    return out


def fig_intensity(means: dict, path: Path) -> None:
    """Overlaid histograms of per-scan mean intensity, one line per class."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 160, 41)
    for c, name in enumerate(config.CLASS_NAMES):
        vals = means[name]
        ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=CLASS_COLORS[c],
                label=f"{name} (n={len(vals)}, mean={vals.mean():.1f})")
    ax.set_xlabel("Mean pixel intensity of scan (0-255, padding excluded)", color=MUTED)
    ax.set_ylabel("Number of scans", color=MUTED)
    ax.set_title("Per-scan mean intensity by class (random train sample)",
                 color=TEXT, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> dict:
    """Run every check, write the JSON report and figures, return the report."""
    torch.set_default_device("cpu")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    shards = {s: _hf_parquet_files(s) for s in ("train", "test")}
    train_cache, train_meta = load_cache("train")
    test_cache, test_meta = load_cache("test")
    metas = {"train": train_meta, "test": test_meta}
    caches = {"train": train_cache, "test": test_cache}
    train_idx, val_idx = patient_split(train_meta)

    print("[1/5] class order ...", flush=True)
    report = {"class_order": check_class_order(shards, metas)}
    print("[2/5] cache correctness ...", flush=True)
    report["cache_check"] = check_cache(shards, caches, metas)
    print("[3/5] patient split ...", flush=True)
    report["patient_split"] = check_patient_split(metas, train_idx, val_idx)
    print("[4/5] image sizes ...", flush=True)
    report["image_sizes_train"] = image_size_distribution(train_meta)
    report["image_sizes_test"] = image_size_distribution(test_meta)
    report["risks"] = size_shortcut_risks(report["image_sizes_train"],
                                          report["image_sizes_test"])
    print("[5/5] val subset ...", flush=True)
    report["val_subset"] = check_val_subset(train_meta)

    print("[figures] ...", flush=True)
    means = intensity_stats(train_cache, train_meta, train_idx)
    report["intensity_train_sample"] = {
        name: {"n": int(len(v)), "mean": round(float(v.mean()), 2),
               "std": round(float(v.std()), 2)} for name, v in means.items()}
    fig_class_distribution(report["patient_split"], report["val_subset"],
                           config.FIGURES_DIR / "data_class_distribution.png")
    fig_samples(shards["test"], test_meta, config.FIGURES_DIR / "data_samples_per_class.png")
    fig_augmentation(train_cache, train_meta,
                     config.FIGURES_DIR / "data_augmentation_examples.png")
    fig_image_sizes({"train": report["image_sizes_train"],
                     "test": report["image_sizes_test"]},
                    config.FIGURES_DIR / "data_image_sizes.png")
    fig_intensity(means, config.FIGURES_DIR / "data_intensity_hist.png")

    report["all_checks_passed"] = all(
        report[k]["passed"] for k in ("class_order", "cache_check", "patient_split",
                                      "val_subset"))
    out = config.METRICS_DIR / "data_report.json"
    out.write_text(json.dumps(report, indent=2))
    for key in ("class_order", "cache_check", "patient_split", "val_subset"):
        print(f"  {key:14s} {'PASS' if report[key]['passed'] else 'FAIL'}")
    for note in report["risks"]:
        print(f"  RISK: {note}")
    print(f"all checks passed: {report['all_checks_passed']} -> {out}")
    return report


if __name__ == "__main__":
    main()
