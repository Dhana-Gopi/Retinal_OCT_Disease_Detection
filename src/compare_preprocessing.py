"""Padding vs direct resize: compare the two ResNet18 runs on the SAME validation split.

Both runs used identical settings (6 epochs x 16,000 class-balanced scans, lr 1e-4,
batch 64, same patient split, same seed); only the resize step differs:

* ``resnet18_pad_finetune`` - pad to square (archived experiment, data/cache_pad_experiment)
* ``resnet18_finetune``     - direct resize to 224x224 (final preprocessing, data/cache)

Each model is evaluated on ALL 10,832 validation scans (473 patients) with the
cache built by its own preprocessing. The decision is made on validation only;
test numbers are written for information and are not used for selection.

Run:  python -m src.compare_preprocessing
Out:  outputs/metrics/preprocessing_comparison.json / .md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src import config
from src.models import load_checkpoint
from src.preprocessing import patient_split, to_model_tensor

RUNS = {
    "pad": ("models/experiments_pad/resnet18_pad_finetune_best.pt", "data/cache_pad_experiment"),
    "stretch": ("models/resnet18_finetune_best.pt", "data/cache"),
}


@torch.no_grad()
def predict(model: torch.nn.Module, images: np.ndarray, rows: np.ndarray,
            batch: int = 128) -> np.ndarray:
    """Predicted class for each cached row (CPU, so the GPU queue is not disturbed)."""
    out = []
    for start in range(0, len(rows), batch):
        x = to_model_tensor(np.array(images[rows[start:start + batch]]))
        out.append(model(x).argmax(1).numpy())
    return np.concatenate(out)


def size_group(meta: pd.DataFrame) -> pd.Series:
    return meta.orig_width.astype(str) + "x" + meta.orig_height.astype(str)


def summarise(y: np.ndarray, p: np.ndarray, sizes: pd.Series) -> dict:
    recalls = recall_score(y, p, average=None, labels=range(config.NUM_CLASSES))
    dme = y == 1
    by_size = {}
    for size, idx in sizes.groupby(sizes.values).groups.items():
        idx = np.asarray(sizes.index.get_indexer(idx))
        if len(idx) >= 20:
            by_size[size] = {"n": int(len(idx)), "acc": float((y[idx] == p[idx]).mean())}
    sq = (sizes.values == "512x512")
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "recall": {c: float(r) for c, r in zip(config.CLASS_NAMES, recalls)},
        "dme_recall_512x512": float((p[dme & sq] == 1).mean()) if (dme & sq).any() else None,
        "dme_recall_other_sizes": float((p[dme & ~sq] == 1).mean()) if (dme & ~sq).any() else None,
        "n_dme_512x512": int((dme & sq).sum()), "n_dme_other": int((dme & ~sq).sum()),
        "accuracy_by_original_size": by_size,
        "confusion_matrix": confusion_matrix(y, p, labels=range(config.NUM_CLASSES)).tolist(),
    }


def main() -> None:
    torch.set_num_threads(6)
    results = {}
    for mode, (ckpt_path, cache) in RUNS.items():
        model, ckpt = load_checkpoint(config.PROJECT_ROOT / ckpt_path, "cpu")
        train_images = np.load(config.PROJECT_ROOT / cache / "train_images.npy", mmap_mode="r")
        test_images = np.load(config.PROJECT_ROOT / cache / "test_images.npy", mmap_mode="r")
        train_meta = pd.read_csv(config.PROJECT_ROOT / cache / "train_meta.csv",
                                 dtype={"patient_id": str})
        test_meta = pd.read_csv(config.PROJECT_ROOT / cache / "test_meta.csv")
        _, val_idx = patient_split(train_meta)           # identical saved split for both
        val_meta = train_meta.iloc[val_idx].reset_index(drop=True)
        print(f"[{mode}] {ckpt['run']} epoch {ckpt['epoch']}: predicting "
              f"{len(val_idx):,} val + {len(test_meta):,} test scans on CPU ...", flush=True)
        val_pred = predict(model, train_images, np.asarray(val_idx))
        test_pred = predict(model, test_images, np.arange(len(test_meta)))
        results[mode] = {
            "run": ckpt["run"], "checkpoint": ckpt_path, "best_epoch": ckpt["epoch"],
            "val_macro_f1_on_3862_subset_during_training": ckpt["val_macro_f1"],
            "validation_full": summarise(val_meta.label.to_numpy(), val_pred, size_group(val_meta)),
            "test_information_only": summarise(test_meta.label.to_numpy(), test_pred,
                                               size_group(test_meta)),
        }
        v = results[mode]["validation_full"]
        print(f"[{mode}] val acc {v['accuracy']:.4f} macro-F1 {v['macro_f1']:.4f} | DME recall "
              f"512x512 {v['dme_recall_512x512']:.3f} vs other {v['dme_recall_other_sizes']:.3f}")

    pad, st = results["pad"]["validation_full"], results["stretch"]["validation_full"]
    gap = lambda r: r["dme_recall_512x512"] - r["dme_recall_other_sizes"]  # noqa: E731
    results["conclusion"] = {
        "val_macro_f1_difference_stretch_minus_pad": st["macro_f1"] - pad["macro_f1"],
        "dme_size_gap_pad": gap(pad), "dme_size_gap_stretch": gap(st),
        "chosen": "stretch" if st["macro_f1"] >= pad["macro_f1"] - 0.005 and gap(st) < gap(pad)
        else "review",
        "rule": "keep direct resize if validation macro-F1 is not worse by >0.5 pt AND the "
                "DME 512x512-vs-other recall gap (size shortcut) is smaller",
    }
    out = config.METRICS_DIR / "preprocessing_comparison.json"
    out.write_text(json.dumps(results, indent=2))

    rows = []
    for mode in ("pad", "stretch"):
        v, t = results[mode]["validation_full"], results[mode]["test_information_only"]
        rows.append(f"| {mode} | {results[mode]['run']} | {results[mode]['best_epoch']} | "
                    f"{v['accuracy']:.4f} | {v['macro_f1']:.4f} | "
                    + " | ".join(f"{v['recall'][c]:.3f}" for c in config.CLASS_NAMES)
                    + f" | {v['dme_recall_512x512']:.3f} / {v['dme_recall_other_sizes']:.3f} | "
                    f"{t['accuracy']:.4f} | {t['macro_f1']:.4f} |")
    md = ("# Padding vs direct resize (ResNet18, identical training settings)\n\n"
          f"Validation = all {pad['n']:,} scans of the saved patient-level split (473 patients). "
          "Test = supplied 1,000 scans, **information only (not used for the decision)**.\n\n"
          "| Resize | Run | Best epoch | Val acc | Val macro-F1 | Val recall CNV | DME | DRUSEN | "
          "NORMAL | Val DME recall 512x512 / other | Test acc | Test macro-F1 |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows)
          + f"\n\nDecision rule: {results['conclusion']['rule']}. "
          f"Outcome: **{results['conclusion']['chosen']}**.\n")
    (config.METRICS_DIR / "preprocessing_comparison.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
