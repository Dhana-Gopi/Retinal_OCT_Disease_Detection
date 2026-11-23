"""Headless UI tests for app.py with streamlit.testing.v1.AppTest.

Run:  python -m tests.test_app
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src import config

APP = str(config.PROJECT_ROOT / "app.py")


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["top_navigation"] = page
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    html = " ".join(m.value for m in at.markdown)
    assert "not a clinical diagnostic tool" in html, "disclaimer missing"
    return at


def test_overview():
    at = run_page("Project Overview")
    labels = {m.label: m.value for m in at.metric}
    assert labels["Training scans (total)"] == "108,309", labels
    assert labels["Test scans"] == "1,000", labels
    html = " ".join(m.value for m in at.markdown)
    assert "Dataset Key Takeaways" in " ".join(s.value for s in at.subheader)
    assert "NORMAL share of training scans" in html
    return labels


def test_performance():
    at = run_page("Model Performance")
    infos = " ".join(i.value for i in at.info)
    subs = " ".join(s.value for s in at.subheader)
    assert "Results pending" in infos or "Model Key Takeaways" in subs
    return "pending" if "Results pending" in infos else "results shown"


def test_try_scan_sample_predict():
    at = run_page("Try a Scan")
    at.radio(key="image_source").set_value("Use a sample test scan").run()
    box = at.selectbox(key="sample_choice")
    drusen = next(p for p in (config.PROJECT_ROOT / "sample_images").glob("DRUSEN_test_*"))
    box.set_value(drusen).run()
    assert at.selectbox(key="sample_choice").value == drusen
    at.button(key="predict").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    labels = {m.label: m.value for m in at.metric}
    assert labels.get("Predicted category") in config.CLASS_NAMES, labels
    assert labels.get("Confidence", "").endswith("%"), labels
    html = " ".join(m.value for m in at.markdown)
    assert "Educational model output" in html
    return drusen.name, labels



SAMPLES = config.PROJECT_ROOT / "sample_images"


def _upload(filename: str, data: bytes, mime: str) -> AppTest:
    at = run_page("Try a Scan")
    at.file_uploader(key="uploader").upload(filename, data, mime).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_upload_predict():
    path = next(SAMPLES.glob("DME_test_*"))
    at = _upload(path.name, path.read_bytes(), "image/jpeg")
    assert not at.error, [e.value for e in at.error]
    assert len(at.image) >= 2, "original + preprocessed preview expected"
    at.button(key="predict").click().run()
    assert not at.exception and not at.error
    labels = {m.label: m.value for m in at.metric}
    assert labels.get("Predicted category") in config.CLASS_NAMES, labels
    return path.name, labels


def test_upload_invalid():
    at = _upload("invalid_not_an_image.jpg", (SAMPLES / "invalid_not_an_image.jpg").read_bytes(),
                 "image/jpeg")
    errors = [e.value for e in at.error]
    assert errors and "Cannot use this file" in errors[0], errors
    assert not [m for m in at.metric if m.label == "Predicted category"]
    return errors[0]


def test_upload_colour_warns_but_predicts():
    path = SAMPLES / "non_oct_colour_example.png"
    at = _upload(path.name, path.read_bytes(), "image/png")
    warnings = [w.value for w in at.warning]
    assert any("strongly coloured" in w for w in warnings), warnings
    at.button(key="predict").click().run()
    assert not at.exception and not at.error
    labels = {m.label: m.value for m in at.metric}
    assert labels.get("Predicted category") in config.CLASS_NAMES, labels
    return warnings[0][:70] + "...", labels


def test_performance_with_results():
    """Render the results view from a temporary eval folder (synthetic numbers)."""
    import json
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    (tmp / "models").mkdir()
    (tmp / "metrics").mkdir()
    (tmp / "models" / "demo_run_history.json").write_text(json.dumps(
        {"run": "demo_run", "status": "complete", "best_val_acc": .9, "best_val_macro_f1": .9}))
    (tmp / "models" / "smoke_x_history.json").write_text(json.dumps(
        {"run": "smoke_x", "status": "complete"}))
    cm = [[240, 5, 4, 1], [6, 236, 2, 6], [30, 3, 205, 12], [0, 2, 8, 240]]
    for run in ("demo_run", "smoke_x"):
        (tmp / "metrics" / f"eval_report_{run}.json").write_text(json.dumps(
            {"run": run, "accuracy": .921, "macro_f1": .92, "confusion_matrix": cm}))
    (tmp / "metrics" / "eval_best_model.json").write_text(json.dumps({"run": "demo_run"}))
    old = config.METRICS_DIR, config.MODELS_DIR
    config.METRICS_DIR, config.MODELS_DIR = tmp / "metrics", tmp / "models"
    try:
        at = run_page("Model Performance")
    finally:
        config.METRICS_DIR, config.MODELS_DIR = old
    html = " ".join(m.value for m in at.markdown)
    assert "Most missed: DRUSEN" in html and "45 of 250" in html, html[-800:]
    assert "Most confused: CNV and DRUSEN" in html and "34 times" in html
    assert at.selectbox[0].options == ["demo_run  (best model)"], at.selectbox[0].options
    return "takeaways rendered; smoke run excluded"


def test_gradcam_expander():
    at = run_page("Try a Scan")
    at.radio(key="image_source").set_value("Use a sample test scan").run()
    at.button(key="predict").click().run()
    at.checkbox(key="gradcam").check().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    infos = [i.value for i in at.info] + [w.value for w in at.warning]
    return {"images": len(at.image), "notes": infos}


TESTS = [test_performance_with_results, test_gradcam_expander, test_overview, test_performance, test_try_scan_sample_predict, test_upload_predict,
         test_upload_invalid, test_upload_colour_warns_but_predicts]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            out = t()
            print(f"PASS {t.__name__}: {out}")
        except Exception as err:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(err).__name__}: {err}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
