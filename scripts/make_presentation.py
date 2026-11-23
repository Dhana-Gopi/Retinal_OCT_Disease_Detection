"""Build the project presentation: reports/Retinal_OCT_Presentation.pptx

Usage (from the project root):
    python scripts/make_presentation.py

The script reads whatever results already exist and fills the slides:
  * outputs/metrics/eval_model_comparison.csv   -> model comparison table
  * outputs/metrics/eval_efficiency.csv          -> speed / size table
  * outputs/metrics/numpy_results.json           -> NumPy fundamentals table
  * outputs/metrics/eval_best_model.json         -> best model + per-class scores
  * outputs/figures/*.png                        -> charts and Grad-CAM images
Anything that does not exist yet is shown as a grey "PENDING" box, so the
script can be re-run at any time and simply gets more complete.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
MET = ROOT / "outputs" / "metrics"
OUT = ROOT / "reports" / "Retinal_OCT_Presentation.pptx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x1A, 0x8C, 0x8C)
DARK = RGBColor(0x26, 0x32, 0x38)
GREY = RGBColor(0x90, 0x9A, 0xA3)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Inches(13.333), Inches(7.5)


# ---------------------------------------------------------------- helpers --
def first_existing(*patterns: str) -> Path | None:
    """Return the first figure matching any of the glob patterns (in order)."""
    for pattern in patterns:
        hits = sorted(FIG.glob(pattern))
        if hits:
            return hits[0]
    return None


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value, pct: bool = False) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value) if value not in (None, "") else "–"
    if pct and v <= 1.0:
        return f"{v * 100:.1f} %"
    return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"


def new_slide(prs: Presentation, title: str, subtitle: str | None = None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])        # blank layout
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tf = bar.text_frame
    tf.margin_left = Inches(0.5)
    p = tf.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(28), True, WHITE
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.0), W, Inches(0.06))
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    accent.line.fill.background()
    if subtitle:
        add_text(slide, subtitle, Inches(0.5), Inches(1.15), Inches(12.3), Inches(0.5),
                 size=14, color=GREY, italic=True)
    footer = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.3))
    fp = footer.text_frame.paragraphs[0]
    fp.text = "Retinal Disease Detection on OCT Images  |  educational prototype, not a diagnostic tool"
    fp.font.size, fp.font.color.rgb = Pt(10), GREY
    return slide


def add_text(slide, text, left, top, width, height, size=16, color=DARK, bold=False,
             italic=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.size, p.font.color.rgb, p.font.bold, p.font.italic = Pt(size), color, bold, italic
    return box


def add_bullets(slide, items, left, top, width, height, size=17):
    """items: list of str or (str, level)."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        text, level = (item, 0) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if level == 0 else "– ") + text
        p.level = level
        p.font.size = Pt(size - 2 * level)
        p.font.color.rgb = DARK if "PENDING" not in text else GREY
        p.space_after = Pt(6)
    return box


def add_pending(slide, label, left, top, width, height):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = GREY
    box.line.dash_style = 4
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"PENDING\n{label}"
    p.alignment = PP_ALIGN.CENTER
    p.font.size, p.font.color.rgb, p.font.italic = Pt(14), GREY, True


def add_image(slide, path: Path | None, left, top, max_w, max_h, label="figure"):
    """Insert an image scaled to fit the box (keeps aspect ratio) or a PENDING box."""
    if path is None or not path.exists():
        add_pending(slide, label, left, top, max_w, max_h)
        return
    pic = slide.shapes.add_picture(str(path), left, top)
    ratio = min(max_w / pic.width, max_h / pic.height)
    pic.width, pic.height = int(pic.width * ratio), int(pic.height * ratio)
    pic.left = int(left + (max_w - pic.width) / 2)
    pic.top = int(top + (max_h - pic.height) / 2)


def add_table(slide, header, rows, left, top, width, row_h=Inches(0.4), size=12):
    table = slide.shapes.add_table(len(rows) + 1, len(header), left, top, width,
                                   Emu(row_h * (len(rows) + 1))).table
    for j, name in enumerate(header):
        cell = table.cell(0, j)
        cell.text = str(name)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        para = cell.text_frame.paragraphs[0]
        para.font.size, para.font.bold, para.font.color.rgb = Pt(size), True, WHITE
    for i, row in enumerate(rows, start=1):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(size)
            para.font.color.rgb = GREY if str(value) == "PENDING" else DARK
    return table


def pick(row: dict, *names: str):
    """Return the first column of ``row`` whose name contains one of ``names``."""
    for name in names:
        for key, value in row.items():
            if key and name in key.lower():
                return value
    return None


# ----------------------------------------------------------------- slides --
def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    comparison = read_csv(MET / "eval_model_comparison.csv")
    efficiency = read_csv(MET / "eval_efficiency.csv")
    numpy_res = read_json(MET / "numpy_results.json")
    best = read_json(MET / "eval_best_model.json")

    # 1. Title -------------------------------------------------------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    add_text(slide, "Retinal Disease Detection", Inches(0.8), Inches(2.0), Inches(11.5),
             Inches(1.0), size=44, color=WHITE, bold=True)
    add_text(slide, "Deep Learning on OCT Images  ·  CNV · DME · DRUSEN · NORMAL",
             Inches(0.8), Inches(3.1), Inches(11.5), Inches(0.6), size=24, color=WHITE)
    add_text(slide, "GUVI Project  ·  PyTorch · Transfer Learning · Grad-CAM · Streamlit",
             Inches(0.8), Inches(4.0), Inches(11.5), Inches(0.5), size=16,
             color=RGBColor(0xB8, 0xD8, 0xD8))
    add_text(slide, "Educational prototype – not a clinical diagnostic tool",
             Inches(0.8), Inches(6.4), Inches(11.5), Inches(0.4), size=12,
             color=RGBColor(0xB8, 0xD8, 0xD8), italic=True)

    # 2. Problem & objectives ----------------------------------------------
    slide = new_slide(prs, "Problem Statement & Objectives")
    add_bullets(slide, [
        "OCT = a light-based cross-section scan of the retina; reading millions by hand is slow",
        "Goal: automatically classify a scan into 4 classes and explain the decision",
        ("CNV – new leaky vessels  ·  DME – diabetic fluid swelling", 1),
        ("DRUSEN – deposits under the retina  ·  NORMAL – healthy", 1),
        "Objectives",
        ("Explore data, build a leakage-free preprocessing pipeline", 1),
        ("Learn fundamentals with a NumPy network", 1),
        ("Custom CNN vs ResNet18, VGG16, MobileNetV2, DenseNet121", 1),
        ("Regularisation, evaluation on unseen test data, Grad-CAM", 1),
        ("Streamlit app: upload a scan → prediction + confidence", 1),
    ], Inches(0.6), Inches(1.4), Inches(12), Inches(5.5))

    # 3. Dataset -------------------------------------------------------------
    slide = new_slide(prs, "Dataset: OCT2017 (Kermany) via Hugging Face",
                      "zacharielegault/Kermany2017-OCT  ·  0 corrupt images  ·  0 duplicates")
    add_table(slide, ["Split", "CNV", "DME", "DRUSEN", "NORMAL", "Total", "Patients"], [
        ["Train", "33,485", "10,213", "7,754", "46,025", "97,477", "4,312"],
        ["Validation", "3,720", "1,135", "862", "5,115", "10,832", "473"],
        ["Test", "250", "250", "250", "250", "1,000", "635"],
    ], Inches(0.5), Inches(1.8), Inches(6.6), size=12)
    add_bullets(slide, [
        "Imbalanced training data: NORMAL 47 %, DRUSEN 8 %",
        "Test set balanced: 250 per class",
        "Sizes vary: 512–1536 px wide, ~496 px tall",
        "All 512×512 scans are DME; test is 94 % 1024×496",
    ], Inches(0.5), Inches(3.8), Inches(6.6), Inches(3), size=15)
    add_image(slide, first_existing("data_class_distribution.png"), Inches(7.3),
              Inches(1.6), Inches(5.6), Inches(5.2), "class distribution chart")

    # 4. Preprocessing -------------------------------------------------------
    slide = new_slide(prs, "Preprocessing, Splitting & Augmentation")
    add_bullets(slide, [
        "Grayscale → direct resize to 224×224 (no padding)",
        ("Padding experiment created a 'no bars = DME' shortcut", 1),
        "Gray copied to 3 channels + ImageNet mean/std (pretrained weights)",
        "One-time uint8 cache (5 GB, memory-mapped) → fast training",
        "Patient-level split (StratifiedGroupKFold): 0 shared patients",
        ("Prevents leakage: same patient's scans look alike", 1),
        "Test set used only once, for final evaluation",
        "Class-balanced sampler for imbalance",
        "GPU augmentation: flip, ±10° rotation, ±10 % zoom, ±5 % shift, ±15 % brightness/contrast",
    ], Inches(0.5), Inches(1.4), Inches(6.9), Inches(5.6), size=15)
    add_image(slide, first_existing("data_augmentation_examples.png", "data_samples*.png"),
              Inches(7.5), Inches(1.4), Inches(5.4), Inches(5.5), "augmentation examples")

    # 5. NumPy fundamentals --------------------------------------------------
    slide = new_slide(prs, "Neural-Network Fundamentals in NumPy",
                      "Forward pass · cross-entropy loss · backpropagation (gradient-checked) · SGD / Momentum / Adam")
    if numpy_res:
        rows = []
        for group, key in (("activation_comparison", "relu"), ("activation_comparison", "sigmoid"),
                           ("activation_comparison", "tanh"), ("optimizer_comparison", "sgd"),
                           ("optimizer_comparison", "momentum"), ("optimizer_comparison", "adam")):
            summ = numpy_res.get(group, {}).get(key, {}).get("summary", {})
            label = key.capitalize() if group.startswith("act") else key.upper() if key == "sgd" else key.capitalize()
            rows.append([label, fmt(summ.get("final_train_acc"), True),
                         fmt(summ.get("best_val_acc"), True)])
        add_table(slide, ["Experiment", "Train acc", "Best val acc"], rows,
                  Inches(0.5), Inches(1.8), Inches(5.2), size=13)
        add_text(slide, "32×32 pixel MLP (1024→128→4): train ≈ 94 % but val ≈ 59 % → "
                 "overfits because it ignores spatial structure → motivates CNNs.",
                 Inches(0.5), Inches(5.0), Inches(5.2), Inches(1.6), size=14)
    else:
        add_pending(slide, "numpy_results.json", Inches(0.5), Inches(1.8), Inches(5.2), Inches(4))
    add_image(slide, first_existing("numpy_activation_comparison.png", "numpy_*.png"),
              Inches(6.0), Inches(1.7), Inches(6.9), Inches(5.2), "NumPy curves")

    # 6. Model development ---------------------------------------------------
    slide = new_slide(prs, "Model Development")
    add_text(slide, "Custom CNN (from scratch, 1.2 M params)", Inches(0.5), Inches(1.3),
             Inches(6), Inches(0.5), size=18, bold=True, color=TEAL)
    add_bullets(slide, [
        "4 blocks: [Conv3×3 → BN → ReLU] ×2 → MaxPool",
        "32 → 64 → 128 → 256 filters",
        "Global average pool → Dropout → FC 128 → FC 4",
        "Ablations: no BatchNorm, no Dropout, no regularisation",
        "Sweep: batch size 32/64/128, lr 1e-3 / 3e-4",
    ], Inches(0.5), Inches(1.8), Inches(6), Inches(4), size=15)
    add_text(slide, "Transfer learning (ImageNet weights)", Inches(6.9), Inches(1.3),
             Inches(6), Inches(0.5), size=18, bold=True, color=TEAL)
    add_table(slide, ["Model", "Params", "Note"], [
        ["ResNet18", "11.2 M", "skip connections"],
        ["VGG16", "134.3 M", "large, slow"],
        ["MobileNetV2", "2.2 M", "light-weight"],
        ["DenseNet121", "7.0 M", "dense connections"],
    ], Inches(6.9), Inches(1.9), Inches(6), size=13)
    add_bullets(slide, [
        "Last layer replaced → 4 classes",
        "Feature extraction (frozen) vs fine-tuning (all layers)",
    ], Inches(6.9), Inches(4.2), Inches(6), Inches(2), size=15)

    # 7. Training strategy ---------------------------------------------------
    slide = new_slide(prs, "Training Strategy on a MacBook Air (MPS GPU)")
    add_table(slide, ["Model", "Train img/s", "Full pass of 97k"], [
        ["Custom CNN", "122", "≈ 13 min"], ["ResNet18", "172", "≈ 9 min"],
        ["MobileNetV2", "170", "≈ 10 min"], ["DenseNet121", "63", "≈ 26 min"],
        ["VGG16", "29", "≈ 56 min"],
    ], Inches(0.5), Inches(1.5), Inches(5.6), size=13)
    add_bullets(slide, [
        "Measured speed first → shortened, class-balanced epochs (16k scans)",
        "Adam + L2 weight decay 1e-4, dropout, ReduceLROnPlateau",
        "Early stopping (patience 3) on validation macro-F1",
        "Best epoch chosen on validation macro-F1 – never on test",
        "Cross-entropy loss, batch size 32–64",
    ], Inches(6.5), Inches(1.5), Inches(6.4), Inches(5), size=16)

    # 8. Results: comparison -------------------------------------------------
    slide = new_slide(prs, "Results: Model Comparison on the Unseen Test Set",
                      "1,000 test scans (250 per class), best-validation checkpoint of each model")
    if comparison:
        rows = []
        for r in comparison[:8]:
            rows.append([pick(r, "run", "model") or "?",
                         fmt(pick(r, "accuracy", "acc"), True),
                         fmt(pick(r, "macro_f1", "f1"), True),
                         fmt(pick(r, "precision"), True),
                         fmt(pick(r, "recall"), True)])
        add_table(slide, ["Model", "Accuracy", "Macro-F1", "Precision", "Recall"], rows,
                  Inches(0.5), Inches(1.7), Inches(6.3), size=12)
    else:
        add_pending(slide, "eval_model_comparison.csv", Inches(0.5), Inches(1.7),
                    Inches(6.3), Inches(4.8))
    add_image(slide, first_existing("eval_model_comparison*.png", "eval_*comparison*.png"),
              Inches(7.0), Inches(1.6), Inches(5.9), Inches(5.2), "comparison chart")

    # 9. Best model ----------------------------------------------------------
    best_name = (best or {}).get("run") or (best or {}).get("model") or "PENDING"
    slide = new_slide(prs, f"Best Model: {best_name}")
    if best and isinstance(best.get("per_class"), dict):
        rows = [[c, fmt(v.get("precision"), True), fmt(v.get("recall"), True),
                 fmt(v.get("f1") or v.get("f1-score"), True)]
                for c, v in best["per_class"].items()]
        add_table(slide, ["Class", "Precision", "Recall", "F1"], rows,
                  Inches(0.5), Inches(1.6), Inches(5.6), size=13)
    else:
        add_pending(slide, "per-class precision / recall / F1", Inches(0.5), Inches(1.6),
                    Inches(5.6), Inches(3))
    add_bullets(slide, ["Saved as models/best_model.pt → used by the app",
                        "Most common confusion: PENDING"],
                Inches(0.5), Inches(5.0), Inches(5.6), Inches(1.8), size=15)
    add_image(slide, first_existing("eval_confusion*best*.png", "eval_confusion*.png"),
              Inches(6.4), Inches(1.4), Inches(6.5), Inches(5.5), "confusion matrix")

    # 10. Curves & regularisation -------------------------------------------
    slide = new_slide(prs, "Training Curves & Regularisation")
    add_image(slide, first_existing("curves_all*.png", "curves_*.png"),
              Inches(0.5), Inches(1.4), Inches(7.4), Inches(5.5), "training curves")
    add_bullets(slide, [
        "Overfitting = train loss ↓ while val loss ↑",
        "BatchNorm: PENDING",
        "Dropout / L2: PENDING",
        "Fine-tune vs feature extraction: PENDING",
        "Early stopping: PENDING",
    ], Inches(8.2), Inches(1.5), Inches(4.8), Inches(5.3), size=15)

    # 11. Grad-CAM -----------------------------------------------------------
    slide = new_slide(prs, "Explainability with Grad-CAM",
                      "Heat-map of the regions that pushed the model toward its prediction")
    add_image(slide, first_existing("gradcam_examples*.png", "gradcam_*.png"),
              Inches(0.5), Inches(1.7), Inches(8.3), Inches(5.2), "Grad-CAM examples")
    add_bullets(slide, [
        "Uses gradients of the last conv layer",
        "Checks the model looks at the retina, not borders or artefacts",
        "Observations: PENDING",
    ], Inches(9.0), Inches(1.8), Inches(4.0), Inches(5), size=15)

    # 12. App ----------------------------------------------------------------
    slide = new_slide(prs, "Streamlit Application")
    add_bullets(slide, [
        "streamlit run app.py → http://localhost:8501",
        "Upload JPEG/PNG OCT scan",
        "Same preprocessing as training",
        "Predicted class + confidence + all 4 probabilities",
        "Grad-CAM overlay",
        "Invalid / corrupted files rejected with a clear message",
        "Disclaimer: educational prototype",
    ], Inches(0.5), Inches(1.5), Inches(5.6), Inches(5.3), size=16)
    add_image(slide, first_existing("app_*.png") or
              next(iter(sorted((ROOT / "reports").glob("app_screenshot*.png"))), None),
              Inches(6.3), Inches(1.4), Inches(6.6), Inches(5.5), "app screenshot")

    # 13. Key findings -------------------------------------------------------
    slide = new_slide(prs, "Key Findings")
    findings = [
        "Patient-level split: 0 patients shared between train, validation and test",
        "NumPy MLP reaches only ≈ 59 % → spatial (CNN) features are essential",
    ]
    if comparison:
        def score(r):
            try:
                return float(pick(r, "macro_f1", "f1") or 0)
            except ValueError:
                return 0.0
        top = max(comparison, key=score)
        findings.append(f"Best test macro-F1: {pick(top, 'run', 'model')} "
                        f"({fmt(pick(top, 'macro_f1', 'f1'), True)}, accuracy "
                        f"{fmt(pick(top, 'accuracy', 'acc'), True)})")
    else:
        findings.append("Best model and test scores: PENDING")
    findings += ["Transfer learning vs custom CNN: PENDING",
                 "Hardest class / most common confusion: PENDING",
                 "Fastest model suitable for CPU deployment: PENDING"]
    add_bullets(slide, findings, Inches(0.6), Inches(1.5), Inches(12), Inches(5.3), size=18)

    # 14. Challenges ---------------------------------------------------------
    slide = new_slide(prs, "Challenges & Solutions")
    add_table(slide, ["Challenge", "Solution"], [
        ["108k JPEGs – slow decoding", "one-time preprocessed, memory-mapped cache"],
        ["Laptop GPU – limited compute", "measured throughput → shortened balanced epochs + early stopping"],
        ["Class imbalance (47 % vs 8 %)", "class-balanced sampler + macro-F1 selection"],
        ["Data leakage (many scans per patient)", "patient-level StratifiedGroupKFold"],
        ["Padding → 'no bars = DME' shortcut", "found per-size; switched to direct resize"],
        ["Grayscale vs RGB pretrained nets", "gray → 3 channels + ImageNet normalisation"],
        ["VGG16 checkpoint 512 MB", "excluded from Git, reproducible from script"],
    ], Inches(0.5), Inches(1.5), Inches(12.3), row_h=Inches(0.55), size=14)

    # 15. Future & limitations ----------------------------------------------
    slide = new_slide(prs, "Limitations & Future Improvements")
    add_text(slide, "Limitations", Inches(0.5), Inches(1.3), Inches(6), Inches(0.5),
             size=18, bold=True, color=TEAL)
    add_bullets(slide, [
        "Shortened training runs",
        "Single dataset / scanner type",
        "Train/test image-size shift + size bias",
        "Small test set (1,000 scans)",
        "Grad-CAM is coarse",
        "Not clinically validated",
    ], Inches(0.5), Inches(1.9), Inches(6), Inches(5), size=16)
    add_text(slide, "Future work", Inches(6.9), Inches(1.3), Inches(6), Inches(0.5),
             size=18, bold=True, color=TEAL)
    add_bullets(slide, [
        "Full-length training on a stronger GPU",
        "EfficientNet / ConvNeXt / Vision Transformers",
        "Ensembles, test-time augmentation",
        "Calibrated confidence + 'refer to doctor' threshold",
        "External validation on other OCT datasets",
        "Online deployment with a light model",
    ], Inches(6.9), Inches(1.9), Inches(6), Inches(5), size=16)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"Saved {out} ({len(Presentation(out).slides)} slides)")
