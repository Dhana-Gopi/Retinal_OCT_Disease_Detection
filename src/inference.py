"""Single-image inference helpers shared by the Streamlit app and the tests.

Everything here re-uses the *training* preprocessing from ``src.preprocessing``
(``preprocess_pil`` + ``to_model_tensor``) so an uploaded image is turned into
exactly the same tensor the model saw during training and evaluation.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src import config
from src.models import load_checkpoint
from src.preprocessing import preprocess_pil, to_model_tensor

# ---- Upload validation limits -------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "MPO"}  # MPO = multi-picture JPEG
MIN_SIDE = 64                       # smaller images carry too little detail
MAX_SIDE = 10_000                   # guard against absurd dimensions
MAX_PIXELS = 50_000_000             # ~50 MP
MAX_BYTES = 25 * 1024 * 1024        # 25 MB

# "Does this look like a grayscale OCT scan?" heuristic thresholds.
SATURATION_WARN = 0.12              # mean HSV saturation (0-1)
COLOURFUL_FRACTION_WARN = 0.10      # fraction of pixels with saturation > 0.25

CLASS_DESCRIPTIONS = {
    "CNV": "Choroidal neovascularisation - abnormal new blood vessels growing under "
           "the retina, typically seen in 'wet' age-related macular degeneration.",
    "DME": "Diabetic macular edema - fluid build-up and swelling in the macula caused "
           "by leaking blood vessels in diabetic retinopathy.",
    "DRUSEN": "Drusen - yellowish deposits of lipids/proteins under the retina, an "
              "early sign of age-related macular degeneration.",
    "NORMAL": "Normal retina - no sign of CNV, DME or drusen in the scan.",
}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def _checkpoint_info(path: Path) -> dict | None:
    """Read the metadata (not used weights) of a checkpoint; None if unreadable."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # half-written file during training, corrupt file, ...
        return None
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        return None
    return {
        "path": str(path),
        "file": path.name,
        "run": ckpt.get("run", path.stem.replace("_best", "")),
        "model_name": ckpt.get("model_name", "?"),
        "mode": ckpt.get("mode", "?"),
        "epoch": ckpt.get("epoch"),
        "val_acc": ckpt.get("val_acc"),
        "val_macro_f1": ckpt.get("val_macro_f1"),
        "preprocessing": ckpt.get("preprocessing", ""),
        # Checkpoints saved before the switch to direct resizing had no key -> "pad".
        "resize_mode": ckpt.get("resize_mode", "pad"),
        "class_names": list(ckpt.get("class_names", [])),
    }


def available_checkpoints(models_dir: Path | str | None = None) -> list[dict]:
    """List usable checkpoints: ``best_model.pt`` (if present) + ``*_best.pt``.

    Smoke-test checkpoints (``smoke_*``) are excluded. Sorted with
    ``best_model.pt`` first, then by validation macro-F1 (highest first).
    """
    models_dir = Path(models_dir or config.MODELS_DIR)
    paths = [p for p in sorted(models_dir.glob("*_best.pt"))
             if not p.name.startswith("smoke_")]
    # The archived pad-to-square experiment is listed last, clearly labelled.
    paths += sorted((models_dir / "experiments_pad").glob("*_best.pt"))
    best = models_dir / "best_model.pt"
    if best.exists():
        paths.insert(0, best)
    infos = [info for p in paths if (info := _checkpoint_info(p)) is not None]
    for info in infos:
        info["is_best_model"] = info["file"] == "best_model.pt"
        info["is_experiment"] = "experiments_pad" in info["path"]
    infos.sort(key=lambda i: (not i["is_best_model"], i["is_experiment"],
                              -(i["val_macro_f1"] or -1.0)))
    return infos


def load_model(path: Path | str, device: str | torch.device = "cpu"):
    """Load a checkpoint -> (model in eval mode, checkpoint dict)."""
    model, ckpt = load_checkpoint(path, device)
    class_names = list(ckpt.get("class_names", config.CLASS_NAMES))
    assert class_names == config.CLASS_NAMES, (
        f"Checkpoint class order {class_names} != config {config.CLASS_NAMES}")
    return model, ckpt


def pick_device(prefer_mps: bool = False) -> torch.device:
    """CPU by default (training owns the GPU); MPS only when asked and available."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_image(model: torch.nn.Module, pil_image: Image.Image,
                  device: str | torch.device = "cpu",
                  class_names: list[str] | None = None,
                  resize_mode: str = config.RESIZE_MODE) -> dict:
    """Run the model on one PIL image using the exact training preprocessing.

    ``resize_mode`` must be the checkpoint's ``resize_mode`` ("stretch" or "pad").
    """
    class_names = list(class_names or config.CLASS_NAMES)
    assert class_names == config.CLASS_NAMES, "class order mismatch"
    gray = np.array(preprocess_pil(pil_image, mode=resize_mode))  # uint8 224x224
    tensor = to_model_tensor(gray).unsqueeze(0).to(device)  # 1x3x224x224
    model.eval()
    probs = torch.softmax(model(tensor).float(), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return {
        "class_name": class_names[idx],
        "class_index": idx,
        "confidence": float(probs[idx]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
        "input_tensor": tensor.cpu(),
        "preprocessed_uint8": gray,
    }


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
def validate_upload(data: bytes | None, filename: str = "") -> Image.Image:
    """Return a decoded PIL image or raise ``ValueError`` with a friendly message."""
    if not data:
        raise ValueError("The uploaded file is empty. Please choose a valid OCT image.")
    if len(data) > MAX_BYTES:
        raise ValueError(f"The file is too large ({len(data) / 1e6:.1f} MB). "
                         f"Please upload an image under {MAX_BYTES // (1024 * 1024)} MB.")
    ext = Path(filename).suffix.lower() if filename else ""
    if filename and ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext or 'none'}'. Allowed types: "
                         + ", ".join(sorted(e.lstrip('.') for e in ALLOWED_EXTENSIONS)) + ".")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
            probe.verify()                      # structural check without decoding
    except (UnidentifiedImageError, Image.DecompressionBombError):
        raise ValueError("This file is not a recognised image. Please upload a "
                         "JPG, PNG, BMP or TIFF retinal OCT scan.") from None
    except Exception:
        raise ValueError("The image file appears to be corrupted and could not be "
                         "read. Please try another file.") from None
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f"Unsupported image format '{fmt}'. Please upload a JPG, "
                         "PNG, BMP or TIFF image.")
    if width > MAX_SIDE or height > MAX_SIDE or width * height > MAX_PIXELS:
        raise ValueError(f"The image is too large ({width}x{height} px). Maximum is "
                         f"{MAX_SIDE} px per side / {MAX_PIXELS // 1_000_000} MP.")
    if min(width, height) < MIN_SIDE:
        raise ValueError(f"The image is too small ({width}x{height} px). OCT scans "
                         f"need at least {MIN_SIDE} px per side to be analysed.")
    try:                                        # full decode (catches truncation)
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise ValueError("The image file appears to be corrupted or truncated and "
                         "could not be decoded. Please try another file.") from None
    return image


def colour_warning(image: Image.Image) -> str | None:
    """Soft check: real OCT B-scans are grayscale. Return a warning text or None."""
    if image.mode in ("L", "LA", "1", "I", "I;16", "F"):
        return None
    small = image.convert("RGB")
    small.thumbnail((256, 256))
    sat = np.asarray(small.convert("HSV"), dtype=np.float32)[..., 1] / 255.0
    mean_sat = float(sat.mean())
    colourful = float((sat > 0.25).mean())
    if mean_sat > SATURATION_WARN or colourful > COLOURFUL_FRACTION_WARN:
        return (f"This image is strongly coloured (mean saturation {mean_sat:.2f}, "
                f"{colourful:.0%} colourful pixels). Retinal OCT scans are normally "
                "grayscale, so this may not be an OCT image and the prediction may "
                "be meaningless.")
    return None


# ---------------------------------------------------------------------------
# Saved-results helpers (used by the app's "Model Performance" view)
# ---------------------------------------------------------------------------
FINISHED_STATUSES = {"complete", "completed", "early_stopped"}


def completed_runs(models_dir: Path | str | None = None) -> dict[str, dict]:
    """{run: history} for runs whose history status is complete/early_stopped."""
    import json

    runs = {}
    for path in sorted(Path(models_dir or config.MODELS_DIR).glob("*_history.json")):
        if path.name.startswith("smoke_"):
            continue
        try:
            hist = json.loads(path.read_text())
        except Exception:
            continue
        if str(hist.get("status", "")).lower() in FINISHED_STATUSES:
            runs[hist.get("run", path.name[: -len("_history.json")])] = hist
    return runs


def parse_eval_report(report: dict, class_names: list[str] | None = None) -> dict:
    """Normalise an ``eval_report_<run>.json`` into accuracy / per-class / confusion matrix.

    Accepts either a ``per_class`` mapping or an sklearn ``classification_report``
    (output_dict=True) mapping keyed by class name.
    """
    class_names = list(class_names or report.get("class_names") or config.CLASS_NAMES)
    cm = report.get("confusion_matrix")
    if isinstance(cm, dict):                      # {"matrix": [[...]]} style
        cm = cm.get("matrix") or cm.get("counts")
    cm = np.asarray(cm, dtype=int) if cm is not None else None
    if cm is not None and cm.shape != (len(class_names), len(class_names)):
        cm = None
    per_class_src = (report.get("per_class") or report.get("classification_report")
                     or report.get("report") or {})
    per_class = {}
    for i, name in enumerate(class_names):
        row = per_class_src.get(name) or per_class_src.get(str(i)) or {}
        entry = {k: row.get(k) for k in ("precision", "recall", "f1-score", "support")}
        if entry["f1-score"] is None:
            entry["f1-score"] = row.get("f1")
        if cm is not None:
            support = int(cm[i].sum())
            entry["support"] = entry["support"] or support
            if entry["recall"] is None and support:
                entry["recall"] = cm[i, i] / support
        per_class[name] = entry
    accuracy = report.get("accuracy", report.get("test_acc"))
    if isinstance(accuracy, dict):
        accuracy = accuracy.get("value")
    if accuracy is None and cm is not None and cm.sum():
        accuracy = float(np.trace(cm) / cm.sum())
    macro_f1 = report.get("macro_f1", report.get("test_macro_f1"))
    if macro_f1 is None and isinstance(per_class_src.get("macro avg"), dict):
        macro_f1 = per_class_src["macro avg"].get("f1-score")
    return {"class_names": class_names, "accuracy": accuracy, "macro_f1": macro_f1,
            "per_class": per_class, "confusion_matrix": cm}


def confusion_takeaways(cm, class_names: list[str] | None = None) -> dict:
    """Most-missed class (lowest recall) and most-confused pair from a confusion matrix.

    Rows = true class, columns = predicted class.
    """
    class_names = list(class_names or config.CLASS_NAMES)
    cm = np.asarray(cm, dtype=int)
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(cm)), where=support > 0)
    missed_idx = int(np.argmin(recall))
    # Symmetric confusion between each unordered pair of classes.
    best = None
    for i in range(len(cm)):
        for j in range(i + 1, len(cm)):
            total = int(cm[i, j] + cm[j, i])
            if best is None or total > best[0]:
                best = (total, i, j)
    total, i, j = best
    return {
        "most_missed": {"class_name": class_names[missed_idx],
                        "false_negatives": int(support[missed_idx] - cm[missed_idx, missed_idx]),
                        "support": int(support[missed_idx]),
                        "recall": float(recall[missed_idx])},
        "most_confused": {"pair": (class_names[i], class_names[j]), "count": total,
                          f"{class_names[i]}->{class_names[j]}": int(cm[i, j]),
                          f"{class_names[j]}->{class_names[i]}": int(cm[j, i]),
                          "a_as_b": int(cm[i, j]), "b_as_a": int(cm[j, i])},
    }
