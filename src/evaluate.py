"""Evaluate every finished model on the unseen 1,000-scan test split.

    python -m src.evaluate                    # everything (cached predictions reused)
    python -m src.evaluate --force            # re-predict every run
    python -m src.evaluate --skip-efficiency  # reuse eval_efficiency.json, no timing
    python -m src.evaluate --include-smoke    # pipeline test only -> */eval_smoke_test/

A run is evaluated when ``models/{run}_best.pt`` exists and its history status
is ``complete`` or ``early_stopped`` (``running`` and ``smoke_*`` runs are
skipped).  Runs are grouped as

* final    - the model comparison (custom CNN, ResNet18 fine-tune/feature-extract,
             VGG16, MobileNetV2, DenseNet121)
* ablation - ``cnn_no_*`` regularisation ablations (compared with custom_cnn_finetune)
* sweep    - ``cnn_bs*_lr*`` batch-size / learning-rate sweep

The *best model* is the final run with the highest **validation** macro-F1;
test metrics are reported next to it but never used for selection.

Outputs (outputs/metrics, outputs/figures):
    eval_predictions_<run>.csv, eval_report_<run>.json, eval_confusion_<run>.png,
    curves_<run>.png, eval_model_comparison.{csv,md,png}, eval_efficiency.{json,csv,md},
    curves_regularisation_ablation.png, eval_regularisation_ablation.{csv,md},
    eval_cnn_sweep.{csv,md,png}, eval_classwise_recall.{csv,png},
    eval_error_analysis.{json,md}, eval_misclassified_<best>.png,
    gradcam_<run>.png, gradcam_model_comparison.png, gradcam_misclassified.png,
    eval_best_model.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (classification_report, confusion_matrix,  # noqa: E402
                             roc_auc_score)

from src import config  # noqa: E402
from src.efficiency import (df_to_markdown, discover_runs, run_efficiency,  # noqa: E402
                            training_running)

CLASSES = config.CLASS_NAMES
GROUP_ORDER = {"final": 0, "ablation": 1, "sweep": 2, "smoke": 3}

# Categorical palette (fixed order, validated light-mode reference palette).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "lines.linewidth": 2,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


# --------------------------------------------------------------------------
# Prediction + metrics
# --------------------------------------------------------------------------
def pick_device(name: str) -> torch.device:
    if name == "auto":
        from src.dataset import get_device

        return get_device()
    return torch.device(name)


_TEST_SETS: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}


def checkpoint_resize_mode(ckpt: dict) -> str:
    """Checkpoints saved before the 'stretch' change have no key -> they used 'pad'."""
    return ckpt.get("resize_mode", "pad")


def peek_resize_mode(path: Path) -> str:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint_resize_mode(ckpt)


def load_test_set(resize_mode: str) -> tuple[np.ndarray, pd.DataFrame]:
    """The 1,000 test scans preprocessed with ``resize_mode`` (uint8 N x 224 x 224) + meta.

    Built from the raw (locally cached) Hugging Face parquet files with
    ``src.preprocessing.preprocess_pil(mode=resize_mode)``, so every checkpoint
    is evaluated with exactly the preprocessing it was trained with, whatever
    mode ``data/cache`` currently holds.  Rows follow ``data/cache/test_meta.csv``.
    """
    if resize_mode in _TEST_SETS:
        return _TEST_SETS[resize_mode]
    from src.preprocessing import _hf_parquet_files, _process_shard, load_cache

    arrays, rows = [], []
    for shard in _hf_parquet_files("test"):
        a, r = _process_shard(str(shard), mode=resize_mode)
        arrays.append(a)
        rows += [x for x in r if not x["corrupt"]]
    images = np.concatenate(arrays)
    meta = pd.DataFrame(rows).drop(columns="corrupt")
    meta["patient_id"] = meta["patient_id"].astype(str)
    # align with the shared cache's row order (used by the test DataLoader / other agents)
    try:
        cache_images, cache_meta = load_cache("test")
        order = pd.Series(np.arange(len(meta)), index=meta.file_name)[cache_meta.file_name].to_numpy()
        images, meta = images[order], meta.iloc[order].reset_index(drop=True)
        if resize_mode == config.RESIZE_MODE and not np.array_equal(images, cache_images):
            print(f"[evaluate] WARNING: data/cache/test_images.npy differs from a fresh "
                  f"'{resize_mode}' preprocessing (cache not rebuilt yet?) - using the fresh one")
    except (FileNotFoundError, KeyError) as err:
        print(f"[evaluate] cache alignment skipped: {err}")
    print(f"[evaluate] test set preprocessed with resize_mode='{resize_mode}': {len(meta)} scans")
    _TEST_SETS[resize_mode] = (images, meta)
    return images, meta


@torch.no_grad()
def predict(model: torch.nn.Module, images: np.ndarray, labels: np.ndarray,
            device: torch.device, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    from src.dataset import prepare_batch

    model.eval()
    probs = []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size]))
        logits = model(prepare_batch(batch, device, augment=False))
        probs.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(probs).numpy(), np.asarray(labels)


def predictions_frame(probs: np.ndarray, labels: np.ndarray, meta: pd.DataFrame,
                      resize_mode: str) -> pd.DataFrame:
    pred = probs.argmax(1)
    df = pd.DataFrame({"file_name": meta["file_name"].values,
                       "patient_id": meta["patient_id"].values,
                       "orig_size": (meta["orig_width"].astype(int).astype(str) + "x"
                                     + meta["orig_height"].astype(int).astype(str)).values,
                       "resize_mode": resize_mode,
                       "true": labels, "pred": pred,
                       "true_name": [CLASSES[i] for i in labels],
                       "pred_name": [CLASSES[i] for i in pred],
                       "confidence": probs.max(1), "correct": pred == labels})
    for i, name in enumerate(CLASSES):
        df[f"prob_{name}"] = probs[:, i]
    return df


def size_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Test accuracy and per-class recall grouped by the scan's original size."""
    rows = []
    for size, g in df.groupby("orig_size"):
        row = {"orig_size": size, "n": int(len(g)), "accuracy": float(g.correct.mean())}
        for c in CLASSES:
            gc_ = g[g.true_name == c]
            row[f"n_{c}"] = int(len(gc_))
            row[f"recall_{c}"] = float(gc_.correct.mean()) if len(gc_) else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def build_report(df: pd.DataFrame, entry: dict) -> dict:
    y, p = df["true"].to_numpy(), df["pred"].to_numpy()
    probs = df[[f"prob_{c}" for c in CLASSES]].to_numpy()
    rep = classification_report(y, p, labels=range(len(CLASSES)), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    try:
        auc = roc_auc_score(y, probs, multi_class="ovr", average=None)
        auc_macro = float(np.mean(auc))
    except ValueError:
        auc, auc_macro = [None] * len(CLASSES), None
    hist = entry["history"]
    return {
        "run": entry["run"], "group": entry["group"], "status": entry["status"],
        "model": hist.get("model"), "mode": hist.get("mode"),
        "checkpoint": str(Path(entry["checkpoint"]).relative_to(config.PROJECT_ROOT)),
        "best_epoch": hist.get("best_epoch"), "val_macro_f1": hist.get("best_val_macro_f1"),
        "val_acc": hist.get("best_val_acc"), "n_test": int(len(df)),
        "accuracy": float(rep["accuracy"]),
        "macro": {k: float(rep["macro avg"][k]) for k in ("precision", "recall", "f1-score")},
        "weighted": {k: float(rep["weighted avg"][k]) for k in ("precision", "recall", "f1-score")},
        "per_class": {c: {"precision": float(rep[c]["precision"]), "recall": float(rep[c]["recall"]),
                          "f1": float(rep[c]["f1-score"]), "support": int(rep[c]["support"]),
                          "roc_auc_ovr": None if auc[i] is None else float(auc[i])}
                      for i, c in enumerate(CLASSES)},
        "macro_roc_auc_ovr": auc_macro,
        "confusion_matrix": confusion_matrix(y, p, labels=range(len(CLASSES))).tolist(),
        "confusion_matrix_labels": "rows = true class, columns = predicted class, order "
                                   + ", ".join(CLASSES),
        "resize_mode": str(df["resize_mode"].iloc[0]) if "resize_mode" in df else None,
        "accuracy_by_orig_size": size_breakdown(df).to_dict("records"),
        "mean_confidence_correct": float(df.loc[df.correct, "confidence"].mean())
        if df.correct.any() else None,
        "mean_confidence_wrong": float(df.loc[~df.correct, "confidence"].mean())
        if (~df.correct).any() else None,
    }


# --------------------------------------------------------------------------
# Figures: confusion matrix, curves
# --------------------------------------------------------------------------
def plot_confusion(report: dict, path: Path) -> None:
    cm = np.array(report["confusion_matrix"])
    pct = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{report['run']} - test set (n={report['n_test']})\n"
                 f"accuracy {report['accuracy']:.3f}, macro-F1 {report['macro']['f1-score']:.3f}")
    fig.savefig(path)
    plt.close(fig)


def fit_diagnosis(history: dict) -> dict:
    """Simple, explainable over/under-fitting heuristics from the epoch curves.

    Note: train metrics are measured *with* augmentation and dropout active,
    so train accuracy slightly below validation accuracy is normal.
    """
    ep = history.get("epochs", [])
    if not ep:
        return {"diagnosis": "no epochs"}
    last, best = ep[-1], max(ep, key=lambda e: e["val_macro_f1"])
    min_val_loss = min(e["val_loss"] for e in ep)
    gap = last["train_acc"] - last["val_acc"]
    val_loss_rise = last["val_loss"] - min_val_loss
    train_falling = len(ep) > 1 and ep[-1]["train_loss"] < ep[0]["train_loss"]
    if best["val_acc"] < 0.80:
        diag = "underfitting (validation accuracy < 0.80)"
    elif gap > 0.05 or (val_loss_rise > 0.10 * min_val_loss and train_falling and len(ep) >= 3):
        diag = "overfitting signs (train >> val or val loss rising)"
    elif history.get("status") == "complete" and best["epoch"] == ep[-1]["epoch"] and len(ep) <= 6:
        diag = "good fit, still improving at last epoch (more epochs may help)"
    else:
        diag = "good fit"
    return {"diagnosis": diag, "final_train_acc": last["train_acc"],
            "final_val_acc": last["val_acc"], "train_val_acc_gap": round(gap, 4),
            "final_train_loss": last["train_loss"], "final_val_loss": last["val_loss"],
            "val_loss_rise_from_min": round(val_loss_rise, 4)}


def plot_curves(entry: dict, path: Path) -> None:
    hist = entry["history"]
    ep = pd.DataFrame(hist["epochs"])
    if ep.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, label in [(axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy")]:
        ax.plot(ep.epoch, ep[f"train_{metric}"], marker="o", ms=5, color=PALETTE[0], label="train")
        ax.plot(ep.epoch, ep[f"val_{metric}"], marker="o", ms=5, color=PALETTE[1], label="validation")
        if metric == "acc":
            ax.plot(ep.epoch, ep.val_macro_f1, ls="--", color=PALETTE[2], label="val macro-F1")
        if hist.get("best_epoch"):
            ax.axvline(hist["best_epoch"], color=INK2, lw=1, ls=":")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_xticks(ep.epoch)
        ax.legend()
    diag = fit_diagnosis(hist)["diagnosis"]
    fig.suptitle(f"{entry['run']} - training curves (dotted line = best epoch)\n{diag}; "
                 "train metrics include augmentation + dropout", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Comparison tables
# --------------------------------------------------------------------------
def comparison_table(reports: dict, entries: dict, eff: dict) -> pd.DataFrame:
    rows = []
    for run, rep in reports.items():
        e, ef = entries[run], eff.get(run, {})
        hist = e["history"]
        rows.append({
            "run": run, "model": rep["model"], "mode": rep["mode"], "group": rep["group"],
            "test_acc": rep["accuracy"], "test_macro_precision": rep["macro"]["precision"],
            "test_macro_recall": rep["macro"]["recall"], "test_macro_f1": rep["macro"]["f1-score"],
            "test_macro_auc": rep["macro_roc_auc_ovr"],
            "val_macro_f1": rep["val_macro_f1"], "best_epoch": rep["best_epoch"],
            "epochs_run": len(hist.get("epochs", [])),
            "params_M": ef.get("params_millions", round(hist.get("total_params", 0) / 1e6, 3)),
            "trainable_params_M": round(hist.get("trainable_params", 0) / 1e6, 3),
            "checkpoint_mb": ef.get("checkpoint_mb",
                                    round(Path(e["checkpoint"]).stat().st_size / 2**20, 2)),
            "gflops": ef.get("gflops_per_image"),
            "train_time_min": round(hist.get("total_train_seconds", sum(
                x["train_seconds"] + x["val_seconds"] for x in hist.get("epochs", []))) / 60, 1),
            "cpu_ms_per_image": ef.get("cpu_latency_ms"),
            "mps_ms_per_image": ef.get("mps_latency_ms"),
            "cpu_throughput_img_s": ef.get("cpu_throughput_img_s"),
            "mps_throughput_img_s": ef.get("mps_throughput_img_s"),
            "fit_diagnosis": fit_diagnosis(hist)["diagnosis"],
            "timing_measured_while_training": ef.get("measured_while_training"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_g"] = df.group.map(GROUP_ORDER)
    return df.sort_values(["_g", "val_macro_f1"], ascending=[True, False]).drop(columns="_g")


def write_table(df: pd.DataFrame, stem: Path, title: str, notes: str = "") -> None:
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(f"# {title}\n\n{notes}\n\n{df_to_markdown(df)}\n")


def plot_comparison(df: pd.DataFrame, path: Path) -> None:
    final = df[df.group.isin(["final", "smoke"])].sort_values("test_macro_f1")
    if final.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(final) + 2.2),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    y = np.arange(len(final))
    for k, (col, lab) in enumerate([("test_acc", "test accuracy"),
                                    ("test_macro_f1", "test macro-F1"),
                                    ("val_macro_f1", "validation macro-F1")]):
        ax.scatter(final[col], y + (k - 1) * 0.18, s=60, color=PALETTE[k], label=lab, zorder=3,
                   edgecolor="white", linewidth=1.5)
    for yi, v in zip(y, final.test_macro_f1):
        ax.text(v, yi + 0.36, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
    ax.set_yticks(y, final.run)
    lo = float(np.nanmin(final[["test_acc", "test_macro_f1", "val_macro_f1"]].to_numpy()))
    ax.set_xlim(max(0.0, lo - 0.03), 1.0)
    ax.set_xlabel("Score (zoomed axis)")
    ax.set_title("Accuracy and macro-F1 per model")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    if final.cpu_ms_per_image.notna().any():
        for i, r in enumerate(final.itertuples()):
            if pd.isna(r.cpu_ms_per_image):
                continue
            ax.scatter(r.cpu_ms_per_image, r.test_macro_f1, s=40 + 6 * r.params_M,
                       color=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolor="white", zorder=3)
            ax.annotate(r.run, (r.cpu_ms_per_image, r.test_macro_f1), fontsize=8,
                        xytext=(6, 4), textcoords="offset points", color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("CPU latency, ms / image (log scale)")
        ax.set_ylabel("Test macro-F1")
        ax.set_title("Accuracy vs CPU cost (marker area ~ parameters)")
        if df.timing_measured_while_training.fillna(False).any():
            ax.text(0.01, 0.01, "timings measured while training was running",
                    transform=ax.transAxes, fontsize=7, color=PALETTE[7])
    else:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


ABLATION_RUNS = ["custom_cnn_finetune", "cnn_no_dropout", "cnn_no_batchnorm",
                 "cnn_no_regularisation"]


def regularisation_ablation(entries: dict, reports: dict, metrics_dir: Path,
                            figures_dir: Path) -> pd.DataFrame | None:
    runs = [r for r in ABLATION_RUNS if r in entries]
    runs += sorted(r for r in entries if entries[r]["group"] == "ablation" and r not in runs)
    if len(runs) < 2:
        print("[ablation] fewer than 2 regularisation runs available - skipped")
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, run in enumerate(runs):
        ep = pd.DataFrame(entries[run]["history"]["epochs"])
        for ax, col in zip(axes.flat, ["train_loss", "val_loss", "train_acc", "val_acc"]):
            ax.plot(ep.epoch, ep[col], marker="o", ms=4, color=PALETTE[i], label=run)
    for ax, title in zip(axes.flat, ["Training loss", "Validation loss",
                                     "Training accuracy", "Validation accuracy"]):
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Custom CNN - effect of Dropout / L2 weight decay / BatchNorm", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "curves_regularisation_ablation.png")
    plt.close(fig)

    rows = []
    for run in runs:
        a = entries[run]["history"].get("args", {})
        rep = reports.get(run, {})
        fd = fit_diagnosis(entries[run]["history"])
        rows.append({"run": run, "dropout": a.get("dropout"), "weight_decay_L2": a.get("weight_decay"),
                     "batchnorm": not a.get("no_bn", False), "lr": a.get("lr"),
                     "batch_size": a.get("batch_size"),
                     "epochs_run": len(entries[run]["history"]["epochs"]),
                     "val_macro_f1": entries[run]["history"].get("best_val_macro_f1"),
                     "test_acc": rep.get("accuracy"),
                     "test_macro_f1": rep.get("macro", {}).get("f1-score"),
                     "final_train_acc": fd.get("final_train_acc"),
                     "final_val_acc": fd.get("final_val_acc"),
                     "train_val_acc_gap": fd.get("train_val_acc_gap"),
                     "val_loss_rise_from_min": fd.get("val_loss_rise_from_min"),
                     "diagnosis": fd["diagnosis"]})
    df = pd.DataFrame(rows)
    write_table(df, metrics_dir / "eval_regularisation_ablation",
                "Custom CNN regularisation ablation",
                "Baseline `custom_cnn_finetune` uses Dropout + L2 (Adam weight_decay) + BatchNorm; "
                "each ablation removes one (or all) of them. Gap = final-epoch train acc - val acc "
                "(train acc is measured with augmentation + dropout on).")
    return df


def cnn_sweep(entries: dict, reports: dict, metrics_dir: Path, figures_dir: Path) -> None:
    runs = sorted(r for r in entries if entries[r]["group"] == "sweep")
    if not runs:
        print("[sweep] no cnn_bs*_lr* runs available - skipped")
        return
    if "custom_cnn_finetune" in entries:
        runs = ["custom_cnn_finetune"] + runs
    rows = []
    for run in runs:
        h = entries[run]["history"]
        a, rep = h.get("args", {}), reports.get(run, {})
        rows.append({"run": run, "batch_size": a.get("batch_size"), "lr": a.get("lr"),
                     "epochs_run": len(h["epochs"]), "best_epoch": h.get("best_epoch"),
                     "val_macro_f1": h.get("best_val_macro_f1"),
                     "test_acc": rep.get("accuracy"),
                     "test_macro_f1": rep.get("macro", {}).get("f1-score"),
                     "train_time_min": round(h.get("total_train_seconds", 0) / 60, 1),
                     "diagnosis": fit_diagnosis(h)["diagnosis"]})
    df = pd.DataFrame(rows)
    write_table(df, metrics_dir / "eval_cnn_sweep", "Custom CNN batch-size / learning-rate sweep",
                "Selection criterion: validation macro-F1 (test shown for reference only).")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, run in enumerate(runs):
        ep = pd.DataFrame(entries[run]["history"]["epochs"])
        a = entries[run]["history"].get("args", {})
        ax.plot(ep.epoch, ep.val_macro_f1, marker="o", ms=4, color=PALETTE[i % len(PALETTE)],
                label=f"{run} (bs {a.get('batch_size')}, lr {a.get('lr')})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Custom CNN: batch size / learning rate sweep")
    ax.legend(fontsize=8)
    fig.savefig(figures_dir / "eval_cnn_sweep.png")
    plt.close(fig)


def classwise_recall(reports: dict, metrics_dir: Path, figures_dir: Path) -> pd.DataFrame:
    rows = []
    for run, rep in reports.items():
        if rep["group"] not in ("final", "smoke"):
            continue
        row = {"run": run}
        for c in CLASSES:
            row[f"recall_{c}"] = rep["per_class"][c]["recall"]
            row[f"precision_{c}"] = rep["per_class"][c]["precision"]
        rec = [rep["per_class"][c]["recall"] for c in CLASSES]
        row["recall_gap_max_min"] = max(rec) - min(rec)
        row["worst_recall_class"] = CLASSES[int(np.argmin(rec))]
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(metrics_dir / "eval_classwise_recall.csv", index=False)
    mat = df[[f"recall_{c}" for c in CLASSES]].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(df) + 1.6))
    ax.imshow(mat, cmap="Blues", vmin=min(0.5, mat.min()), vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > 0.85 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(df)), df.run)
    ax.set_title("Per-class recall on the test set (class-wise bias)")
    fig.savefig(figures_dir / "eval_classwise_recall.png")
    plt.close(fig)
    return df


# --------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------
CONFUSION_NOTES = {
    frozenset({"CNV", "DRUSEN"}): "drusen and early CNV both deform the RPE band; small drusen "
                                  "vs. fibrovascular PED are hard to separate at 224 px",
    frozenset({"DRUSEN", "NORMAL"}): "small/sparse drusen produce only subtle RPE undulations "
                                     "that resemble a normal retina after downscaling",
    frozenset({"CNV", "DME"}): "both show intraretinal/subretinal fluid; fluid location "
                               "(above vs. below the RPE) is the key cue",
    frozenset({"DME", "NORMAL"}): "mild DME with small cysts can look almost normal",
    frozenset({"CNV", "NORMAL"}): "scans with small lesions or poor signal quality",
    frozenset({"DME", "DRUSEN"}): "irregular outer retina in both classes",
}


def error_analysis(df: pd.DataFrame, report: dict, metrics_dir: Path) -> dict:
    wrong = df[~df.correct]
    pairs = (wrong.groupby(["true_name", "pred_name"]).size().rename("count").reset_index()
             .sort_values("count", ascending=False))
    pairs["pct_of_true_class"] = pairs.apply(
        lambda r: r["count"] / max(1, (df.true_name == r.true_name).sum()), axis=1)
    recalls = {c: report["per_class"][c]["recall"] for c in CLASSES}
    precisions = {c: report["per_class"][c]["precision"] for c in CLASSES}
    pred_counts = df.pred_name.value_counts().reindex(CLASSES, fill_value=0)
    true_counts = df.true_name.value_counts().reindex(CLASSES, fill_value=0)
    over = {c: int(pred_counts[c] - true_counts[c]) for c in CLASSES}
    worst = min(recalls, key=recalls.get)
    best = max(recalls, key=recalls.get)
    # patient-level concentration of errors (does one patient dominate?)
    per_patient = wrong.groupby("patient_id").size().sort_values(ascending=False)
    top_pairs = pairs.head(5).to_dict("records")
    suggestions = []
    for rec in top_pairs[:3]:
        note = CONFUSION_NOTES.get(frozenset({rec["true_name"], rec["pred_name"]}))
        if note:
            suggestions.append(f"{rec['true_name']} -> {rec['pred_name']} ({rec['count']} scans): "
                               f"{note}.")
    suggestions += [
        f"Class-wise bias: recall ranges from {recalls[worst]:.3f} ({worst}) to "
        f"{recalls[best]:.3f} ({best}); raise the weight of {worst} in the loss or "
        f"oversample it further, and tune per-class decision thresholds on validation data.",
        "Higher input resolution (e.g. 384-512 px) or crops centred on the RPE/fovea would keep "
        "the fine drusen / small-fluid detail that 224 px downsampling removes.",
        "Train longer on the full training split (runs here used shortened 16k-scan epochs) "
        "with a cosine learning-rate schedule; add test-time augmentation (h-flip).",
        "Ensemble the two or three best architectures (soft-voting of probabilities).",
        "Review high-confidence errors with a clinician: some may be label noise "
        "(Kermany labels are per-scan, graded by non-uniform expert pools).",
        "Use the softmax confidence to flag low-confidence scans for human review "
        "instead of auto-classifying them.",
    ]
    out = {
        "run": report["run"], "n_test": int(len(df)), "n_misclassified": int(len(wrong)),
        "error_rate": float(len(wrong) / len(df)),
        "most_confused_pairs": [{k: (float(v) if isinstance(v, (np.floating, float)) else
                                     int(v) if isinstance(v, (np.integer,)) else v)
                                 for k, v in r.items()} for r in pairs.to_dict("records")],
        "per_class_recall": recalls, "per_class_precision": precisions,
        "recall_gap_max_min": float(recalls[best] - recalls[worst]),
        "worst_recall_class": worst, "best_recall_class": best,
        "predicted_minus_true_counts": over,
        "mean_confidence_correct": report["mean_confidence_correct"],
        "mean_confidence_wrong": report["mean_confidence_wrong"],
        "high_confidence_errors_gt_0.9": int((wrong.confidence > 0.9).sum()),
        "patients_with_errors": int(per_patient.size),
        "max_errors_single_patient": int(per_patient.iloc[0]) if per_patient.size else 0,
        "suggested_improvements": suggestions,
    }
    (metrics_dir / "eval_error_analysis.json").write_text(json.dumps(out, indent=2))
    pairs_md = pairs.assign(pct_of_true_class=pairs.pct_of_true_class.map("{:.1%}".format))
    rec_df = pd.DataFrame({"class": CLASSES, "recall": [recalls[c] for c in CLASSES],
                           "precision": [precisions[c] for c in CLASSES],
                           "true_count": true_counts.values, "predicted_count": pred_counts.values})
    md = [f"# Error analysis - best model `{report['run']}`\n",
          f"Test scans: {len(df)}, misclassified: {len(wrong)} ({len(wrong) / len(df):.1%}). "
          f"Mean confidence correct {out['mean_confidence_correct'] or 0:.3f} vs wrong "
          f"{out['mean_confidence_wrong'] or 0:.3f}; errors with confidence > 0.9: "
          f"{out['high_confidence_errors_gt_0.9']}. Errors spread over "
          f"{out['patients_with_errors']} patients (max {out['max_errors_single_patient']} "
          "from one patient).\n",
          "## Most-confused pairs (true -> predicted)\n", df_to_markdown(pairs_md), "",
          "## Per-class recall / precision (class-wise bias)\n", df_to_markdown(rec_df), "",
          f"Recall gap (max - min): **{out['recall_gap_max_min']:.3f}**, weakest class "
          f"**{worst}**. Over-/under-prediction vs. true counts: "
          + ", ".join(f"{c} {v:+d}" for c, v in over.items()) + ".\n",
          "## Suggested improvements\n", *[f"- {s}" for s in suggestions], ""]
    (metrics_dir / "eval_error_analysis.md").write_text("\n".join(md))
    return out


def plot_misclassified(df: pd.DataFrame, images: np.ndarray, run: str, path: Path,
                       n: int = 16) -> None:
    wrong = df[~df.correct].sort_values("confidence", ascending=False).head(n)
    if wrong.empty:
        return
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (idx, r) in zip(axes.flat, wrong.iterrows()):
        ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"true {r.true_name} -> pred {r.pred_name}\nconf {r.confidence:.2f} | "
                     f"{r.file_name}", fontsize=8)
    fig.suptitle(f"{run}: most confident misclassified test scans "
                 f"({(~df.correct).sum()} errors / {len(df)})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Grad-CAM figures (CPU by default so training on the GPU is not disturbed)
# --------------------------------------------------------------------------
def gradcam_sample_indices(meta: pd.DataFrame, per_class: int = 2, seed: int = config.SEED
                           ) -> list[int]:
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(CLASSES)):
        pool = np.flatnonzero(meta["label"].to_numpy() == c)
        idx += sorted(rng.choice(pool, per_class, replace=False).tolist())
    return idx


def _cams(model, name, images, indices, device, target=None):
    from src.gradcam import compute_gradcam
    from src.preprocessing import to_model_tensor

    out = []
    for k, i in enumerate(indices):
        x = to_model_tensor(np.array(images[i]))[None].to(device)
        t = None if target is None else int(target[k])
        out.append(compute_gradcam(model, name, x, t))
    return out


def gradcam_figures(final_runs: list[str], entries: dict, preds: dict, images: np.ndarray,
                    meta: pd.DataFrame, best_run: str | None, figures_dir: Path,
                    device: str = "cpu") -> None:
    from src.gradcam import overlay_heatmap
    from src.models import load_checkpoint

    sample = gradcam_sample_indices(meta)
    overlays = {}
    for run in final_runs:
        model, ckpt = load_checkpoint(entries[run]["checkpoint"], device)
        cams = _cams(model, ckpt["model_name"], images, sample, device)
        overlays[run] = [overlay_heatmap(np.array(images[i]), h) for i, (h, _) in zip(sample, cams)]
        fig, axes = plt.subplots(4, 4, figsize=(11, 11.8))
        for k, (i, (h, p)) in enumerate(zip(sample, cams)):
            r, c = divmod(k, 2)
            a0, a1 = axes[r, 2 * c], axes[r, 2 * c + 1]
            a0.imshow(images[i], cmap="gray", vmin=0, vmax=255)
            a0.set_title(f"original - true {CLASSES[meta.label.iloc[i]]}", fontsize=9)
            a1.imshow(overlays[run][k])
            pred = int(np.argmax(p))
            a1.set_title(f"Grad-CAM - pred {CLASSES[pred]} ({p[pred]:.2f})", fontsize=9,
                         color=INK if pred == meta.label.iloc[i] else PALETTE[7])
            a0.axis("off")
            a1.axis("off")
        fig.suptitle(f"Grad-CAM: {run} ({ckpt['model_name']}) - 2 test scans per class\n"
                     "bright/warm = regions that increased the predicted-class score",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(figures_dir / f"gradcam_{run}.png")
        plt.close(fig)
        del model

    if len(overlays) >= 2:
        runs = list(overlays)
        fig, axes = plt.subplots(len(sample), len(runs) + 1,
                                 figsize=(2.2 * (len(runs) + 1), 2.35 * len(sample)), squeeze=False)
        for k, i in enumerate(sample):
            axes[k, 0].imshow(images[i], cmap="gray", vmin=0, vmax=255)
            axes[k, 0].set_ylabel(CLASSES[meta.label.iloc[i]])
            for j, run in enumerate(runs):
                axes[k, j + 1].imshow(overlays[run][k])
                pred = int(preds[run].pred.iloc[i])
                axes[k, j + 1].set_title(
                    f"pred {CLASSES[pred]}", fontsize=8,
                    color=INK if pred == meta.label.iloc[i] else PALETTE[7])
            for ax in axes[k]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
        for j, title in enumerate(["original"] + runs):
            axes[0, j].set_title(title + ("\n" + axes[0, j].get_title() if j else ""), fontsize=8)
        fig.suptitle("Grad-CAM comparison across models (same test scans)", fontsize=11)
        fig.tight_layout()
        fig.savefig(figures_dir / "gradcam_model_comparison.png")
        plt.close(fig)

    if best_run is None:
        return
    df = preds[best_run]
    wrong = df[~df.correct].sort_values("confidence", ascending=False).head(8)
    if wrong.empty:
        return
    model, ckpt = load_checkpoint(entries[best_run]["checkpoint"], device)
    idx = wrong.index.tolist()
    cam_pred = _cams(model, ckpt["model_name"], images, idx, device, wrong.pred.to_numpy())
    cam_true = _cams(model, ckpt["model_name"], images, idx, device, wrong.true.to_numpy())
    rows = int(np.ceil(len(idx) / 2))
    fig, axes = plt.subplots(rows, 6, figsize=(15, 2.9 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (i, r) in enumerate(wrong.iterrows()):
        rr, cc = divmod(k, 2)
        a = axes[rr, 3 * cc: 3 * cc + 3]
        a[0].imshow(images[i], cmap="gray", vmin=0, vmax=255)
        a[0].set_title(f"true {r.true_name}, pred {r.pred_name} ({r.confidence:.2f})", fontsize=8)
        a[1].imshow(overlay_heatmap(np.array(images[i]), cam_pred[k][0]))
        a[1].set_title(f"CAM for predicted {r.pred_name}", fontsize=8)
        a[2].imshow(overlay_heatmap(np.array(images[i]), cam_true[k][0]))
        a[2].set_title(f"CAM for true {r.true_name}", fontsize=8)
    fig.suptitle(f"{best_run}: Grad-CAM on the most confident misclassified test scans",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "gradcam_misclassified.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def select_best(entries: dict, reports: dict) -> tuple[str | None, str]:
    for groups in (("final",), ("smoke",)):
        cands = [r for r in reports if entries[r]["group"] in groups]
        if cands:
            best = max(cands, key=lambda r: entries[r]["history"].get("best_val_macro_f1") or -1)
            return best, (f"highest validation macro-F1 "
                          f"({entries[best]['history'].get('best_val_macro_f1'):.4f}) among "
                          f"{len(cands)} {'/'.join(groups)} runs ({', '.join(sorted(cands))}); "
                          "test metrics were not used for selection")
    return None, "no candidate runs"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--include-smoke", action="store_true",
                   help="TESTING ONLY: include smoke_* runs; outputs go to */eval_smoke_test/")
    p.add_argument("--runs", nargs="*", help="restrict to these run names")
    p.add_argument("--force", action="store_true", help="re-predict even if cached CSV is fresh")
    p.add_argument("--device", default="auto", help="device for test-set prediction")
    p.add_argument("--gradcam-device", default="cpu")
    p.add_argument("--skip-efficiency", action="store_true",
                   help="do not time models; reuse eval_efficiency.json if present")
    p.add_argument("--skip-gradcam", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)

    t_start = time.time()
    metrics_dir, figures_dir = config.METRICS_DIR, config.FIGURES_DIR
    if args.include_smoke:
        metrics_dir, figures_dir = metrics_dir / "eval_smoke_test", figures_dir / "eval_smoke_test"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(include_smoke=args.include_smoke)
    if args.runs:
        runs = [r for r in runs if r["run"] in set(args.runs)]
    if not runs:
        print("[evaluate] no finished runs to evaluate")
        return
    entries = {r["run"]: r for r in runs}
    print(f"[evaluate] runs: {', '.join(entries)}")

    from src.dataset import make_loaders
    from src.models import load_checkpoint
    from src.preprocessing import load_cache

    device = pick_device(args.device)
    test_images, test_meta = load_cache("test")
    test_loader = None
    preds, reports = {}, {}
    for run, e in entries.items():
        csv = metrics_dir / f"eval_predictions_{run}.csv"
        if (not args.force and csv.exists()
                and csv.stat().st_mtime > Path(e["checkpoint"]).stat().st_mtime):
            df = pd.read_csv(csv, dtype={"patient_id": str})
            print(f"[evaluate] {run}: cached predictions")
        else:
            if test_loader is None:
                test_loader = make_loaders(batch_size=args.batch_size)[2]
            model, _ = load_checkpoint(e["checkpoint"], device)
            t0 = time.time()
            probs, labels = predict(model, test_loader, device)
            assert len(labels) == len(test_meta) and (labels == test_meta.label.to_numpy()).all()
            df = predictions_frame(probs, labels, test_meta)
            df.to_csv(csv, index=False)
            print(f"[evaluate] {run}: predicted {len(df)} test scans on {device} "
                  f"in {time.time() - t0:.1f}s", flush=True)
            del model
        preds[run] = df
        rep = build_report(df, e)
        reports[run] = rep
        (metrics_dir / f"eval_report_{run}.json").write_text(json.dumps(rep, indent=2))
        plot_confusion(rep, figures_dir / f"eval_confusion_{run}.png")
        plot_curves(e, figures_dir / f"curves_{run}.png")
        print(f"[evaluate] {run}: test acc {rep['accuracy']:.4f} "
              f"macro-F1 {rep['macro']['f1-score']:.4f} (val macro-F1 {rep['val_macro_f1']:.4f})")
    if device.type == "mps":
        torch.mps.empty_cache()

    # ---- efficiency (timed fairly only when idle; flagged otherwise) ----
    if args.skip_efficiency:
        path = metrics_dir / "eval_efficiency.json"
        eff = json.loads(path.read_text()).get("runs", {}) if path.exists() else {}
    else:
        eff = run_efficiency(runs, metrics_dir)

    comp = comparison_table(reports, entries, eff)
    busy_note = ("**Note:** some latency numbers were measured while a training job was running "
                 "(`timing_measured_while_training`); rerun `python -m src.efficiency "
                 "--only-if-idle && python -m src.evaluate --skip-efficiency` when idle.\n"
                 if comp.timing_measured_while_training.fillna(False).any() else "")
    write_table(comp, metrics_dir / "eval_model_comparison", "Model comparison (test set, n=1000)",
                "Groups: final = model comparison, ablation = regularisation ablation, "
                "sweep = batch-size/lr sweep. Model selection uses validation macro-F1 only. "
                "Latency = median single-image forward pass (ms). " + busy_note)
    plot_comparison(comp, figures_dir / "eval_model_comparison.png")
    regularisation_ablation(entries, reports, metrics_dir, figures_dir)
    cnn_sweep(entries, reports, metrics_dir, figures_dir)
    classwise_recall(reports, metrics_dir, figures_dir)

    best, reason = select_best(entries, reports)
    if best:
        rep = reports[best]
        (metrics_dir / "eval_best_model.json").write_text(json.dumps({
            "run": best, "checkpoint": rep["checkpoint"],
            "checkpoint_abs": str(entries[best]["checkpoint"]),
            "model": rep["model"], "mode": rep["mode"], "reason": reason,
            "val_macro_f1": rep["val_macro_f1"], "test_accuracy": rep["accuracy"],
            "test_macro_f1": rep["macro"]["f1-score"],
            "smoke_test_only": entries[best]["group"] == "smoke",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))
        error_analysis(preds[best], rep, metrics_dir)
        plot_misclassified(preds[best], test_images, best,
                           figures_dir / f"eval_misclassified_{best}.png")
        print(f"[evaluate] best model: {best} - {reason}")

    if not args.skip_gradcam:
        final_runs = [r for r in comp.run if entries[r]["group"] in ("final", "smoke")]
        gradcam_figures(final_runs, entries, preds, test_images, test_meta, best, figures_dir,
                        args.gradcam_device)
    if training_running():
        print("[evaluate] note: a training job is still running - rerun at the end.")
    print(f"[evaluate] done in {time.time() - t_start:.0f}s -> {metrics_dir}, {figures_dir}")


if __name__ == "__main__":
    main()
