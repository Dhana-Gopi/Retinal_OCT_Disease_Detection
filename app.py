"""Retinal OCT Disease Detection | White & Yellow edition.

Run:  streamlit run app.py
Educational prototype - NOT a clinical diagnostic tool.

Views
  1. Project Overview   - dataset size, classes, class balance, example scans
  2. Model Performance  - saved test-set results of completed training runs
  3. Try a Scan         - upload an OCT scan and get an educational model output

All predictions use src/inference.py, which re-uses the exact training
preprocessing (src/preprocessing.py) and the class order in src/config.py.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import config
from src import inference as inf

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "sample_images"
YELLOW = "#F4B400"
GOLD = "#C78B00"
INK = "#202735"
GRAY = "#657080"
LOW_CONFIDENCE = 0.60
DISCLAIMER = ("<b>Educational prototype - not a clinical diagnostic tool.</b> "
              "This app was built for a student project. Its outputs are model "
              "estimates on research data, may be wrong, and must never be used to "
              "diagnose, treat or rule out any eye disease. Please consult a qualified "
              "ophthalmologist for any medical concern.")

st.set_page_config(page_title="Retinal OCT Disease Detection", page_icon="👁️",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown('''<style>
:root {color-scheme: light;}
.stApp {background:#FFFFFF;color:#222222}
[data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #EDE8D9}
.block-container {padding-top:1.5rem;max-width:1480px;padding-left:2rem;padding-right:2rem;padding-bottom:3rem}
@media (max-width: 640px) {.block-container {padding-left:1rem;padding-right:1rem}}
h1,h2,h3 {color:#202735 !important;letter-spacing:-.025em}
h1 {font-size:2rem !important}
[data-testid="stMetric"] {border:1px solid #EDE8D9;border-radius:13px;padding:15px 17px;box-shadow:0 3px 12px #17171708}
[data-testid="stMetricLabel"] {color:#657080}
[data-testid="stMetricValue"] {color:#202735;font-weight:750}
.stButton>button[kind="primary"],button[kind="primary"] {background:#F4B400;color:#202735;border:0;font-weight:700}
.stButton>button[kind="primary"]:hover {background:#E5A500;color:#202735}
[data-testid="stDataFrame"] {border:1px solid #EDE8D9;border-radius:10px}
.note {background:#FFFBE6;border-left:4px solid #F4B400;padding:13px 17px;border-radius:8px;margin:12px 0 18px;color:#202735}
.eyebrow {color:#9A7400;text-transform:uppercase;letter-spacing:.12em;font-size:.73rem;font-weight:800}
hr {border-color:#EDE8D9 !important}
.insight-card{background:#FFFBE6;border-left:4px solid #F2C300;border-radius:7px;padding:10px 12px;margin:6px 0 11px;font-size:13px;line-height:1.5;color:#222}
.section-title{font-weight:850;font-size:1.15rem;color:#202735;border-bottom:2px solid #F2C300;padding-bottom:4px;margin:14px 0 9px}
[data-testid="stMetric"]{background:#FFFBE6 !important;border-top:0 !important;border-left:4px solid #F2C300 !important}
.chart-explainer{border-top:1px solid #F1E9D0;padding:11px 4px 2px;color:#596273;font-size:12.5px;line-height:1.55}
.chart-explainer b{color:#8D6A00;display:block;font-size:11px;text-transform:uppercase;letter-spacing:.045em;margin-bottom:3px}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:15px !important;box-shadow:0 4px 22px rgba(32,39,53,.045)}
[data-testid="stHorizontalBlock"]{row-gap:1.4rem}
[data-testid="stRadio"] [role="radiogroup"]{justify-content:center;gap:6px !important;flex-wrap:wrap;background:#fff;border-bottom:1px solid #EDE8D9;padding-bottom:12px}
[data-testid="stRadio"] label{font-weight:700;padding:9px 12px;border-radius:9px;white-space:nowrap}
[data-testid="stRadio"] label:has(input:checked){background:#FFF4C2;color:#202735}
.class-card{background:#FFFBE6;border-left:4px solid #F2C300;border-radius:7px;padding:9px 12px;margin:5px 0;font-size:13px;line-height:1.45}
.class-card b{color:#202735}
.result-label{color:#9A7400;text-transform:uppercase;letter-spacing:.1em;font-size:.75rem;font-weight:800;margin-bottom:4px}
</style>''', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Presentation helpers (patterns reused from the reference app)
# ---------------------------------------------------------------------------
def note(html: str) -> None:
    st.markdown(f'<div class="note">{html}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def heading(title: str, subtitle: str) -> None:
    st.markdown('<div class="eyebrow">Retinal OCT · deep learning</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def metric_row(items) -> None:
    cols = st.columns(len(items))
    for col, (name, value, helptext) in zip(cols, items):
        col.metric(name, value, help=helptext)


def chart(altair_chart, title: str, meaning: str | None = None) -> None:
    """Chart + 'What this chart denotes' explainer kept together in one card."""
    with st.container(border=True):
        st.markdown(f"**{escape(title)}**")
        st.altair_chart(altair_chart, width="stretch")
        if meaning:
            st.markdown(f'<div class="chart-explainer"><b>What this chart denotes</b>'
                        f'<div>{escape(meaning)}</div></div>', unsafe_allow_html=True)


def insights_panel(title: str, points, subtitle: str | None = None) -> None:
    """Key-takeaways rail: bold heading + one plain-English sentence each."""
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Key takeaways</div>', unsafe_allow_html=True)
        st.subheader(title)
        if subtitle:
            st.caption(subtitle)
        for head, body in points:
            st.markdown('<div class="insight-card"><strong>' + escape(str(head))
                        + '</strong><div style="margin-top:5px">' + escape(str(body))
                        + '</div></div>', unsafe_allow_html=True)


def bar_chart(df: pd.DataFrame, x: str, y: str, fmt: str = ",", height: int = 280,
              color: str = YELLOW, x_title: str = "", y_title: str = "",
              domain: tuple | None = None) -> alt.Chart:
    scale = alt.Scale(domain=list(domain)) if domain else alt.Undefined
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:N", sort=None, title=x_title, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y}:Q", title=y_title, scale=scale))
    bars = base.mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    text = base.mark_text(dy=-7, color=INK, fontWeight="bold").encode(
        text=alt.Text(f"{y}:Q", format=fmt))
    return (bars + text).properties(height=height).configure_view(stroke=None) \
        .configure_axis(grid=False, labelColor=INK, titleColor=GRAY)


def class_card(name: str) -> None:
    st.markdown(f'<div class="class-card"><b>{name}</b> - '
                f'{escape(inf.CLASS_DESCRIPTIONS[name])}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def dataset_stats(mtimes: tuple) -> dict | None:
    """Counts computed from the cached metadata + saved patient split."""
    train_csv = config.CACHE_DIR / "train_meta.csv"
    test_csv = config.CACHE_DIR / "test_meta.csv"
    split_json = config.CACHE_DIR / "patient_split.json"
    if not (train_csv.exists() and test_csv.exists()):
        return None
    train = pd.read_csv(train_csv, dtype={"patient_id": str})
    test = pd.read_csv(test_csv, dtype={"patient_id": str})
    stats = {
        "n_train_total": len(train), "n_test": len(test),
        "train_counts": train["label"].value_counts().reindex(range(4), fill_value=0).tolist(),
        "test_counts": test["label"].value_counts().reindex(range(4), fill_value=0).tolist(),
        "train_patients": train["patient_id"].nunique(),
        "test_patients": test["patient_id"].nunique(),
        "patient_overlap": len(set(train.patient_id) & set(test.patient_id)),
    }
    if split_json.exists():
        split = json.loads(split_json.read_text())
        tr, va = split["train_idx"], split["val_idx"]
        stats.update(n_train=len(tr), n_val=len(va),
                     val_patients=train["patient_id"].iloc[va].nunique())
    return stats


def sample_files() -> dict[str, list[Path]]:
    """Original test JPEGs exported by the data agent, grouped by true class."""
    out = {name: [] for name in config.CLASS_NAMES}
    if SAMPLES_DIR.exists():
        for path in sorted(SAMPLES_DIR.iterdir()):
            prefix = path.name.split("_test_")[0]
            if prefix in out and path.suffix.lower() in inf.ALLOWED_EXTENSIONS:
                out[prefix].append(path)
    return out


@st.cache_data(show_spinner=False)
def cached_test_example(class_index: int) -> np.ndarray | None:
    """Fallback example scan from the preprocessed test cache."""
    try:
        images = np.load(config.CACHE_DIR / "test_images.npy", mmap_mode="r")
        meta = pd.read_csv(config.CACHE_DIR / "test_meta.csv")
        row = int(meta.index[meta["label"] == class_index][0])
        return np.array(images[row])
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def list_checkpoints(key: tuple) -> list[dict]:
    return inf.available_checkpoints()


@st.cache_resource(show_spinner="Loading model ...")
def get_model(path: str, device: str, mtime_key: float):
    return inf.load_model(path, device)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def eval_results(key: tuple) -> dict:
    """Saved evaluation-agent outputs, restricted to completed runs."""
    done = inf.completed_runs()
    metrics = config.METRICS_DIR
    comparison = None
    csv_path = metrics / "eval_model_comparison.csv"
    if csv_path.exists():
        try:
            comparison = pd.read_csv(csv_path)
            run_col = next((c for c in comparison.columns if c.lower() in ("run", "run_name")), None)
            if run_col is not None:
                comparison = comparison[comparison[run_col].astype(str).isin(done)
                                        & ~comparison[run_col].astype(str).str.startswith("smoke_")]
        except Exception:
            comparison = None
    reports = {}
    for path in sorted(metrics.glob("eval_report_*.json")):
        run = path.stem[len("eval_report_"):]
        if run not in done:
            continue
        raw = read_json(path)
        if raw is None:
            continue
        try:
            reports[run] = inf.parse_eval_report(raw)
        except Exception:
            continue
    best = read_json(metrics / "eval_best_model.json") or {}
    best_run = best.get("run") or best.get("best_run") or best.get("run_name")
    return {"completed": done, "comparison": comparison, "reports": reports,
            "best_run": best_run if best_run in reports else None, "best_raw": best}


def results_key() -> tuple:
    paths = list(config.METRICS_DIR.glob("eval_*")) + list(config.MODELS_DIR.glob("*_history.json"))
    return (str(config.METRICS_DIR), str(config.MODELS_DIR),
            *sorted((p.name, _mtime(p)) for p in paths))


# ---------------------------------------------------------------------------
# Page header + navigation
# ---------------------------------------------------------------------------
st.markdown('<h1 style="text-align:center;font-weight:900;color:#202735;margin-bottom:0">'
            'RETINAL OCT DISEASE DETECTION</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#666;margin:2px 0 12px">Optical coherence '
            'tomography • CNV · DME · DRUSEN · NORMAL • Deep learning • Educational prototype</p>',
            unsafe_allow_html=True)
PAGES = ["Project Overview", "Model Performance", "Try a Scan"]
page = st.radio("Navigation", PAGES, horizontal=True, label_visibility="collapsed",
                key="top_navigation")
note(DISCLAIMER)


# ---------------------------------------------------------------------------
# 1. Project Overview
# ---------------------------------------------------------------------------
def page_overview() -> None:
    heading("Project Overview", "Kermany 2018 retinal OCT dataset · four diagnostic categories "
            "· patient-level train/validation split")
    stats = dataset_stats((_mtime(config.CACHE_DIR / "train_meta.csv"),
                           _mtime(config.CACHE_DIR / "test_meta.csv"),
                           _mtime(config.CACHE_DIR / "patient_split.json")))
    if stats is None:
        st.info("Dataset statistics are not available yet (data/cache has not been built).")
        return
    metric_row([
        ("Training scans (total)", f"{stats['n_train_total']:,}", "All scans in the supplied training split"),
        ("Used for training", f"{stats.get('n_train', 0):,}", "Patient-level split: training part"),
        ("Used for validation", f"{stats.get('n_val', 0):,}", "Patient-level split: held-out patients for model selection"),
        ("Test scans", f"{stats['n_test']:,}", "Supplied test split, never used for training"),
    ])
    left, right = st.columns([3.2, 1.25], gap="large")
    names = config.CLASS_NAMES
    train_counts, test_counts = stats["train_counts"], stats["test_counts"]
    total = max(sum(train_counts), 1)
    with left:
        a, b = st.columns(2, gap="medium")
        with a:
            df = pd.DataFrame({"Class": names, "Scans": train_counts})
            chart(bar_chart(df, "Class", "Scans", y_title="Training scans"),
                  "Training scans per category",
                  "Shows how many training scans belong to each disease category; the "
                  "bars are uneven, so the classes are imbalanced.")
        with b:
            df = pd.DataFrame({"Class": names, "Scans": test_counts})
            chart(bar_chart(df, "Class", "Scans", y_title="Test scans", color=GOLD),
                  "Test scans per category",
                  "Shows the held-out test set, which has the same number of scans in "
                  "every category, so each class counts equally in the final score.")
        section("The four categories")
        c1, c2 = st.columns(2, gap="medium")
        for i, name in enumerate(names):
            with (c1 if i % 2 == 0 else c2):
                class_card(name)
        section("One example scan per category")
        st.caption("Original grayscale test-set scans; the label above each image is the "
                   "ground-truth category from the dataset.")
        samples = sample_files()
        cols = st.columns(4, gap="small")
        for i, (col, name) in enumerate(zip(cols, names)):
            with col:
                if samples[name]:
                    path = samples[name][0]
                    col.image(Image.open(path).convert("L"), caption=f"{name} · {path.name.split('_test_')[-1]}",
                              width="stretch")
                else:
                    arr = cached_test_example(i)
                    if arr is not None:
                        col.image(arr, caption=f"{name} (preprocessed 224×224)",
                                  width="stretch", clamp=True)
                    else:
                        col.info(f"No {name} example available.")
    with right:
        normal_share = train_counts[3] / total
        rare_idx = int(np.argmin(train_counts))
        common_idx = int(np.argmax(train_counts))
        ratio = train_counts[common_idx] / max(train_counts[rare_idx], 1)
        points = [
            ("NORMAL share of training scans",
             f"{normal_share:.1%} of the {total:,} training scans are NORMAL, so normal "
             "retinas are a large part of what the model sees."),
            (f"{names[rare_idx]} is the rarest class",
             f"Only {train_counts[rare_idx]:,} training scans ({train_counts[rare_idx] / total:.1%}) "
             f"are {names[rare_idx]}, about {ratio:.1f}× fewer than {names[common_idx]}."),
            ("Why recall is checked per class",
             f"With this imbalance a model could score a high overall accuracy while still "
             f"missing many {names[rare_idx]} scans, so we report recall for every class."),
            ("Balanced test set",
             f"The test set has {', '.join(f'{c:,}' for c in test_counts)} scans per class "
             f"({'/'.join(names)}), so no class can dominate the test accuracy."),
        ]
        if "n_val" in stats:
            points.append(("No patient leakage",
                           f"Validation uses {stats['val_patients']:,} patients who never appear in "
                           f"training, so scores reflect new patients rather than memorised ones."))
        insights_panel("Dataset Key Takeaways", points,
                       f"{stats['train_patients']:,} training patients · {stats['test_patients']:,} test patients")


# ---------------------------------------------------------------------------
# 2. Model Performance
# ---------------------------------------------------------------------------
def _fmt_pct(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.1%}"


def confusion_chart(cm: np.ndarray, names: list[str]) -> alt.Chart:
    rows = [{"True class": names[i], "Predicted class": names[j], "Scans": int(cm[i, j])}
            for i in range(len(names)) for j in range(len(names))]
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Predicted class:N", sort=names, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("True class:N", sort=names))
    rect = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("Scans:Q", scale=alt.Scale(range=["#FFFBE6", YELLOW, "#9C6900"]),
                        legend=None),
        tooltip=["True class", "Predicted class", "Scans"])
    vmax = max(int(cm.max()), 1)
    text = base.mark_text(fontSize=15, fontWeight="bold").encode(
        text="Scans:Q",
        color=alt.condition(alt.datum.Scans > vmax * 0.6, alt.value("white"), alt.value(INK)))
    return (rect + text).properties(height=330).configure_view(stroke=None) \
        .configure_axis(labelColor=INK, titleColor=GRAY, labelFontSize=12)


def page_performance() -> None:
    heading("Model Performance", "Saved test-set results of completed training runs only "
            "(no live retraining, smoke tests excluded)")
    res = eval_results(results_key())
    done, reports = res["completed"], res["reports"]
    if not reports:
        st.info("**Results pending.** The test-set evaluation files "
                "(outputs/metrics/eval_report_<run>.json) for completed runs are not "
                "available yet. This page will fill in automatically once the evaluation "
                "step has saved them - no numbers are shown until then.")
        if done:
            section("Completed training runs (validation metrics from training history)")
            st.dataframe(pd.DataFrame([{
                "Run": run, "Model": h.get("model"), "Mode": h.get("mode"),
                "Status": h.get("status"), "Best epoch": h.get("best_epoch"),
                "Val accuracy": _fmt_pct(h.get("best_val_acc")),
                "Val macro-F1": _fmt_pct(h.get("best_val_macro_f1"))}
                for run, h in done.items()]), hide_index=True, width="stretch")
        else:
            st.caption("No training run has finished yet.")
        return

    # ---- Comparison of completed runs --------------------------------------
    comp_rows = [{"Run": run, "Test accuracy": r["accuracy"], "Test macro-F1": r["macro_f1"]}
                 for run, r in reports.items()]
    comp = pd.DataFrame(comp_rows)
    left, right = st.columns([3.2, 1.25], gap="large")
    with left:
        section("Model comparison (test set)")
        plot_df = comp.dropna(subset=["Test accuracy"]).copy()
        if not plot_df.empty:
            plot_df["Accuracy (%)"] = plot_df["Test accuracy"].astype(float) * 100
            plot_df = plot_df.sort_values("Accuracy (%)", ascending=False)
            chart(bar_chart(plot_df, "Run", "Accuracy (%)", fmt=".1f", y_title="Test accuracy (%)",
                            domain=(0, 100)),
                  "Test accuracy by model",
                  "Each bar is one completed model scored on the same 1,000 unseen test "
                  "scans; taller means more scans classified correctly.")
        table = res["comparison"] if res["comparison"] is not None and not res["comparison"].empty else comp
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Source: outputs/metrics/eval_model_comparison.csv and eval_report_<run>.json "
                   "(written by the evaluation step).")

        # ---- Confusion matrix of the selected run --------------------------
        runs = list(reports)
        default = runs.index(res["best_run"]) if res["best_run"] in runs else 0
        run = st.selectbox("Run to inspect", runs, index=default,
                           format_func=lambda r: f"{r}  (best model)" if r == res["best_run"] else r)
        rep = reports[run]
        names = rep["class_names"]
        metric_row([("Test accuracy", _fmt_pct(rep["accuracy"]), "Share of the test scans classified correctly"),
                    ("Test macro-F1", _fmt_pct(rep["macro_f1"]), "Average F1 over the four classes (each class weighted equally)"),
                    ("Test scans", f"{int(rep['confusion_matrix'].sum()):,}" if rep["confusion_matrix"] is not None else "-",
                     "Scans in the evaluated test set")])
        cm = rep["confusion_matrix"]
        a, b = st.columns([1.2, 1], gap="medium")
        with a:
            if cm is not None:
                chart(confusion_chart(cm, names), f"Confusion matrix · {run}",
                      "Rows are the true category and columns the model's prediction; the "
                      "diagonal counts correct answers and every other cell is a mistake.")
            else:
                st.info("This report has no confusion matrix.")
        with b:
            pc = pd.DataFrame([{"Class": n, "Recall": v["recall"], "Precision": v["precision"],
                                "F1": v["f1-score"]} for n, v in rep["per_class"].items()])
            if pc["Recall"].notna().any():
                rec = pc.assign(**{"Recall (%)": pc["Recall"].astype(float) * 100})
                chart(bar_chart(rec, "Class", "Recall (%)", fmt=".1f", y_title="Recall (%)",
                                domain=(0, 100), height=250),
                      "Recall per category",
                      "Recall is the share of real scans of each category that the model "
                      "found; a low bar means that disease is often missed.")
            st.dataframe(pc.style.format({c: "{:.1%}" for c in ["Recall", "Precision", "F1"]},
                                         na_rep="-"),
                         hide_index=True, width="stretch")
    with right:
        if cm is not None:
            tk = inf.confusion_takeaways(cm, names)
            mm, mc = tk["most_missed"], tk["most_confused"]
            a_name, b_name = mc["pair"]
            points = [
                (f"Most missed: {mm['class_name']}",
                 f"The model missed {mm['false_negatives']} of {mm['support']} {mm['class_name']} "
                 f"scans (recall {mm['recall']:.1%}), the lowest of the four classes."),
                (f"Most confused: {a_name} and {b_name}",
                 f"These two were mixed up {mc['count']} times ({a_name}→{b_name}: {mc['a_as_b']}, "
                 f"{b_name}→{a_name}: {mc['b_as_a']}), more than any other pair."),
                ("Overall accuracy",
                 f"{_fmt_pct(rep['accuracy'])} of the {int(cm.sum()):,} unseen test scans were "
                 f"classified correctly by {run}."),
                ("How to read this",
                 "These are results on research data from one dataset, so they do not show how "
                 "the model would perform in a real clinic."),
            ]
            insights_panel("Model Key Takeaways", points, f"Run: {run}")
        else:
            st.info("Takeaways need a confusion matrix, which this report does not contain.")


# ---------------------------------------------------------------------------
# 3. Try a Scan
# ---------------------------------------------------------------------------
def render_gradcam(model, ckpt, result) -> None:
    try:
        from src.gradcam import compute_gradcam, overlay_heatmap
    except Exception:
        st.info("Grad-CAM is not available in this build yet (src/gradcam.py not found).")
        return
    try:
        import torch
        with torch.enable_grad():
            device = next(model.parameters()).device
            heatmap, _ = compute_gradcam(model, ckpt["model_name"],
                                         result["input_tensor"].to(device),
                                         target_class=result["class_index"])
        overlay = overlay_heatmap(result["preprocessed_uint8"], np.asarray(heatmap))
        g1, g2 = st.columns(2)
        g1.image(result["preprocessed_uint8"], caption="Model input (224×224)", width="stretch")
        g2.image(overlay, caption=f"Grad-CAM for {result['class_name']}", width="stretch")
        st.caption("Warm colours mark the regions that most increased the score for the "
                   "predicted class. It is a rough visual explanation, not a lesion map.")
    except Exception:
        st.warning("Grad-CAM could not be computed for this model/image.")


def page_try_scan() -> None:
    heading("Try a Scan", "Upload a retinal OCT B-scan · the saved model returns an educational output")
    ckpts = list_checkpoints((str(config.MODELS_DIR), *sorted((p.name, _mtime(p)) for p in config.MODELS_DIR.glob("**/*.pt"))))
    if not ckpts:
        st.warning("No trained model checkpoint is available yet (models/*_best.pt). "
                   "Please train a model first.")
        return
    left, right = st.columns([2.2, 1.25], gap="large")
    with right:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Model</div>', unsafe_allow_html=True)
            labels = {c["path"]: (f"{c['run']} (best model)" if c["is_best_model"] else c["run"])
                      + (" [padding experiment]" if c.get("is_experiment") else "")
                      + (f" · val F1 {c['val_macro_f1']:.3f}" if c["val_macro_f1"] is not None else "")
                      for c in ckpts}
            path = st.selectbox("Saved model", list(labels), format_func=labels.get, key="model_path")
            info = next(c for c in ckpts if c["path"] == path)
            use_mps = st.toggle("Use Apple GPU (MPS)", value=False,
                                help="CPU is fast enough for one image and is the default.")
            st.caption(f"Architecture: **{info['model_name']}** ({info['mode']}) · epoch {info['epoch']}")
            st.caption(f"Validation accuracy {_fmt_pct(info['val_acc'])} · macro-F1 {_fmt_pct(info['val_macro_f1'])}")
            with st.expander("Preprocessing"):
                st.caption(info["preprocessing"] or "grayscale → pad to 224×224 → ImageNet normalise")
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Categories</div>', unsafe_allow_html=True)
            for name in config.CLASS_NAMES:
                class_card(name)

    with left:
        source = st.radio("Image source", ["Upload my own scan", "Use a sample test scan"],
                          horizontal=True, key="image_source")
        data, filename = None, ""
        if source == "Upload my own scan":
            upload = st.file_uploader("Upload a retinal OCT image",
                                      type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                                      key="uploader",
                                      help="Grayscale OCT B-scans (e.g. 1024×496) work best. Max 25 MB.")
            if upload is not None:
                data, filename = upload.getvalue(), upload.name
        else:
            samples = [p for files in sample_files().values() for p in files]
            if not samples:
                st.info("No sample scans found in sample_images/.")
            else:
                chosen = st.selectbox("Sample scan (true class shown in the name)", samples,
                                      format_func=lambda p: p.name, key="sample_choice")
                data, filename = chosen.read_bytes(), chosen.name
        if data is None:
            st.caption("Upload a JPG, PNG, BMP or TIFF OCT scan to begin.")
            return
        try:
            image = inf.validate_upload(data, filename)
        except ValueError as err:
            st.error(f"**Cannot use this file.** {err}")
            return
        warning = inf.colour_warning(image)
        if warning:
            st.warning(warning)

        gray_preview = image.convert("L")
        model_input = inf.preprocess_pil(image, mode=info["resize_mode"])
        p1, p2 = st.columns([1.6, 1], gap="medium")
        p1.image(gray_preview, caption=f"Uploaded scan · {image.size[0]}×{image.size[1]} px (grayscale)",
                 width="stretch")
        p2.image(model_input, caption=("Model input · 224×224 (direct resize, as in training)"
                                       if info["resize_mode"] == "stretch" else
                                       "Model input · 224×224 (aspect kept, black padding)"),
                 width="stretch")

        key = (filename, len(data), path, use_mps)
        if st.button("Predict class", type="primary", width="stretch", key="predict"):
            try:
                device = str(inf.pick_device(use_mps))
                model, ckpt = get_model(path, device, _mtime(Path(path)))
                result = inf.predict_image(model, image, device, ckpt.get("class_names"),
                                           resize_mode=ckpt.get("resize_mode", "pad"))
                st.session_state["prediction"] = (key, result, info)
            except Exception:
                st.session_state.pop("prediction", None)
                st.error("Sorry, the model could not analyse this image. Please try another "
                         "image or a different saved model.")
        saved = st.session_state.get("prediction")
        if not saved or saved[0] != key:
            return
        _, result, info = saved
        with st.container(border=True):
            st.markdown('<div class="result-label">Educational model output - not a diagnosis</div>',
                        unsafe_allow_html=True)
            metric_row([("Predicted category", result["class_name"], "Class with the highest model probability"),
                        ("Confidence", f"{result['confidence']:.1%}", "Softmax probability of the predicted class"),
                        ("Model", info["run"], f"{info['model_name']} · {info['file']}")])
            class_card(result["class_name"])
            if result["confidence"] < LOW_CONFIDENCE:
                st.warning(f"Low confidence ({result['confidence']:.1%} < {LOW_CONFIDENCE:.0%}). The "
                           "model is unsure between categories; treat this output with extra caution.")
            probs = pd.DataFrame({"Class": list(result["probabilities"]),
                                  "Probability (%)": [p * 100 for p in result["probabilities"].values()]})
            chart(bar_chart(probs, "Class", "Probability (%)", fmt=".1f", y_title="Probability (%)",
                            domain=(0, 100), height=240),
                  "Model probability for each category",
                  "Shows how the model split its belief across the four categories; the "
                  "tallest bar is the prediction and the bars add up to 100%.")
            st.markdown(f'<div class="note" style="font-size:13px">{DISCLAIMER}</div>',
                        unsafe_allow_html=True)
        with st.expander("Optional: Grad-CAM explanation"):
            if st.checkbox("Compute Grad-CAM heatmap", key="gradcam"):
                try:
                    model, ckpt = get_model(path, str(inf.pick_device(use_mps)), _mtime(Path(path)))
                    render_gradcam(model, ckpt, result)
                except Exception:
                    st.warning("Grad-CAM could not be computed.")


# ---------------------------------------------------------------------------
# Router - every page is wrapped so users never see a stack trace
# ---------------------------------------------------------------------------
try:
    {"Project Overview": page_overview, "Model Performance": page_performance,
     "Try a Scan": page_try_scan}[page]()
except Exception:
    st.error("Something went wrong while building this page. Please refresh, or try "
             "another view.")

st.divider()
st.caption("Retinal OCT Disease Detection · GUVI student project · Educational prototype - "
           "not for clinical use · Data: Kermany et al. 2018 (CC BY 4.0)")
