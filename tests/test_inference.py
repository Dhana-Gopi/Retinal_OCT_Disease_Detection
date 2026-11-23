"""Tests for src/inference.py (plain asserts).

Run:  python -m tests.test_inference      (or: pytest tests/)
Uses the real ResNet18 checkpoint when its training history says it finished,
otherwise the smoke-test checkpoint (fine for wiring; predictions meaningless).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src import config
from src import inference as inf
from src.preprocessing import to_model_tensor

SAMPLES = config.PROJECT_ROOT / "sample_images"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pick_checkpoint() -> Path:
    hist = config.MODELS_DIR / "resnet18_finetune_history.json"
    real = config.MODELS_DIR / "resnet18_finetune_best.pt"
    try:
        status = json.loads(hist.read_text()).get("status")
    except Exception:
        status = None
    if real.exists() and status in inf.FINISHED_STATUSES:
        return real
    return config.MODELS_DIR / "smoke_resnet18_best.pt"


_MODEL = None


def model_and_ckpt():
    global _MODEL
    if _MODEL is None:
        _MODEL = inf.load_model(pick_checkpoint(), "cpu")
    return _MODEL


def original_test_jpegs() -> list[Path]:
    """Original test JPEGs: sample_images/ (data agent) or tests/fixtures/ fallback."""
    files = sorted(SAMPLES.glob("*_test_*.jp*g")) if SAMPLES.exists() else []
    if files:
        return files
    files = sorted(FIXTURES.glob("*_test_*.jp*g")) if FIXTURES.exists() else []
    if files:
        return files
    # Fallback: export 4 originals (one per class) from the local HF test parquet.
    import pyarrow.parquet as pq
    from src.preprocessing import _hf_parquet_files

    FIXTURES.mkdir(parents=True, exist_ok=True)
    wanted = set(config.CLASS_NAMES)
    for shard in _hf_parquet_files("test"):
        for item in pq.read_table(shard, columns=["image"]).column("image").to_pylist():
            cls = item["path"].split("-")[0]
            if cls in wanted:
                (FIXTURES / f"{cls}_test_{item['path']}").write_bytes(item["bytes"])
                wanted.discard(cls)
        if not wanted:
            break
    return sorted(FIXTURES.glob("*_test_*.jp*g"))


# ---------------------------------------------------------------------------
def test_preprocessing_parity():
    """predict_image(original JPEG) == model(cached training-pipeline array)."""
    model, _ = model_and_ckpt()
    images = np.load(config.CACHE_DIR / "test_images.npy", mmap_mode="r")
    meta = pd.read_csv(config.CACHE_DIR / "test_meta.csv")
    row_of = {name: i for i, name in enumerate(meta["file_name"])}
    files = original_test_jpegs()
    assert files, "no original test JPEGs found"
    for path in files:
        orig_name = path.name.split("_test_", 1)[1]
        row = row_of[orig_name]
        result = inf.predict_image(model, Image.open(path), "cpu")
        cached = np.array(images[row])
        assert np.array_equal(result["preprocessed_uint8"], cached), f"pixel mismatch {path.name}"
        with torch.no_grad():
            ref = torch.softmax(model(to_model_tensor(cached).unsqueeze(0)), 1)[0].numpy()
        got = np.array([result["probabilities"][n] for n in config.CLASS_NAMES])
        assert np.allclose(got, ref, atol=1e-4), f"prob mismatch {path.name}: {got} vs {ref}"
        assert result["input_tensor"].shape == (1, 3, 224, 224)
        assert abs(sum(result["probabilities"].values()) - 1) < 1e-4


def test_class_mapping():
    model, ckpt = model_and_ckpt()
    assert ckpt["class_names"] == config.CLASS_NAMES == ["CNV", "DME", "DRUSEN", "NORMAL"]
    path = original_test_jpegs()[0]
    r = inf.predict_image(model, Image.open(path), "cpu")
    assert config.CLASS_NAMES[r["class_index"]] == r["class_name"]
    assert r["confidence"] == max(r["probabilities"].values())
    assert list(r["probabilities"]) == config.CLASS_NAMES


def _png_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _expect_error(data, name, fragment):
    try:
        inf.validate_upload(data, name)
    except ValueError as err:
        assert fragment.lower() in str(err).lower(), f"{name}: {err}"
        return
    raise AssertionError(f"{name}: expected ValueError")


def test_invalid_inputs():
    good = _png_bytes(Image.new("L", (300, 200), 128))
    assert inf.validate_upload(good, "scan.png").size == (300, 200)
    _expect_error(b"", "empty.png", "empty")
    _expect_error(None, "none.png", "empty")
    _expect_error(b"hello, I am text", "fake.jpg", "not a recognised image")
    bad = SAMPLES / "invalid_not_an_image.jpg"
    if bad.exists():
        _expect_error(bad.read_bytes(), bad.name, "not a recognised image")
    _expect_error(good, "scan.gif", "unsupported file type")
    _expect_error(good, "notes.txt", "unsupported file type")
    _expect_error(_png_bytes(Image.new("L", (40, 300))), "tiny.png", "too small")
    _expect_error(_png_bytes(Image.new("L", (12000, 70))), "huge.png", "too large")
    jpeg = _png_bytes(Image.new("L", (300, 200), 90), "JPEG")
    _expect_error(jpeg[: len(jpeg) // 2], "cut.jpg", "corrupted")
    gif = _png_bytes(Image.new("L", (300, 200)), "GIF")
    _expect_error(gif, "renamed.png", "unsupported image format")


def test_colour_warning():
    path = original_test_jpegs()[0]
    assert inf.colour_warning(Image.open(path)) is None, "real OCT scan should not warn"
    gray_rgb = Image.open(path).convert("RGB")
    assert inf.colour_warning(gray_rgb) is None, "grayscale saved as RGB should not warn"
    colour = SAMPLES / "non_oct_colour_example.png"
    img = Image.open(colour) if colour.exists() else Image.new("RGB", (300, 300), (200, 30, 30))
    assert inf.colour_warning(img) is not None, "colour image should warn"
    # Still predictable after the warning.
    model, _ = model_and_ckpt()
    assert inf.predict_image(model, img, "cpu")["class_name"] in config.CLASS_NAMES


def test_available_checkpoints():
    ckpts = inf.available_checkpoints()
    assert all(not c["file"].startswith("smoke_") for c in ckpts)
    f1s = [c["val_macro_f1"] for c in ckpts if not c["is_best_model"]]
    assert f1s == sorted(f1s, reverse=True)


def test_confusion_takeaways():
    cm = np.array([[240, 5, 4, 1],
                   [6, 236, 2, 6],
                   [30, 3, 205, 12],
                   [0, 2, 8, 240]])
    tk = inf.confusion_takeaways(cm)
    assert tk["most_missed"] == {"class_name": "DRUSEN", "false_negatives": 45,
                                 "support": 250, "recall": 0.82}
    assert tk["most_confused"]["pair"] == ("CNV", "DRUSEN")
    assert tk["most_confused"]["count"] == 34
    parsed = inf.parse_eval_report({"accuracy": 0.921, "confusion_matrix": cm.tolist()})
    assert parsed["per_class"]["DRUSEN"]["recall"] == 0.82
    sk = inf.parse_eval_report({"classification_report": {
        "CNV": {"precision": .9, "recall": .96, "f1-score": .93, "support": 250},
        "macro avg": {"f1-score": .91}}, "confusion_matrix": cm.tolist()})
    assert sk["macro_f1"] == .91 and sk["per_class"]["CNV"]["precision"] == .9
    assert abs(sk["accuracy"] - np.trace(cm) / cm.sum()) < 1e-12


TESTS = [test_preprocessing_parity, test_class_mapping, test_invalid_inputs,
         test_colour_warning, test_available_checkpoints, test_confusion_takeaways]

if __name__ == "__main__":
    print(f"checkpoint: {pick_checkpoint().name}")
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as err:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(err).__name__}: {err}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
