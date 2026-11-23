"""Append the end-to-end viva walkthrough to OCT_dataset.ipynb (lead-owned).

The user's original learning cells (everything before the marker cell) are kept
exactly as they are, including their outputs. Re-running this script replaces
only the walkthrough part.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "OCT_dataset.ipynb"
MARKER = "<!-- VIVA-WALKTHROUGH-START -->"

SAVED = "🟨 **Loaded from the saved training run** (nothing is retrained here)."
COMPUTED = "🟦 **Computed live in this notebook** when you run the cell."

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(f"""
{MARKER}
# **Part 2 — End-to-end walkthrough for the viva**

Everything above this cell is my original exploration (kept unchanged).
From here on, the notebook follows the project's real code in `src/`, step by step:

1. Patient-level split → 2. Preprocessing → 3. Batch creation → 4. Model creation →
5. Training loop → 6. Best-checkpoint selection → 7. Evaluation → 8. One prediction

**How to read the outputs**

| Tag | Meaning |
|---|---|
| 🟨 Saved run | Read from files written by the training script (`models/…`, `outputs/…`) |
| 🟦 Computed here | Calculated live by the cell when you run it |

**Run All is safe:** it uses the saved weights. Retraining is an *optional* cell (Section 5b)
that is switched off by default.
""")

code("""
# Setup: make the project's own modules (the src/ folder) importable.
import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():          # in case the notebook runs from another folder
    PROJECT_ROOT = Path("/Users/dhanalakshmi/Documents/Retinal_OCT_Project")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from src import config


def show_code(function):
    \"\"\"Display the real source code of a project function.\"\"\"
    display(Markdown(f"```python\\n{inspect.getsource(function)}\\n```"))


print("Project folder:", PROJECT_ROOT)
print("Class order   :", config.CLASS_NAMES)
print("Image size    :", config.IMAGE_SIZE, "| resize mode:", config.RESIZE_MODE)
""")

# ---------------------------------------------------------------- 1. split
md("""
## 1. Patient-level train / validation split

**Problem.** One patient usually has *many* scans. If some scans of a patient are in training and
others in validation, the model can recognise the *patient* (eye shape, scanner noise) instead of
the *disease*, and validation accuracy looks better than it really is. This is called **data leakage**.

**Solution.** Each file name contains a patient ID (`CNV-1016042-1.jpeg` → patient `1016042`).
`patient_split` uses `StratifiedGroupKFold`:
* **Group** = patient, so a patient is entirely in train *or* in validation.
* **Stratified** = each side keeps roughly the same class mix.
* 1 of 10 folds (≈10 % of patients) becomes validation; `seed = 42` makes it reproducible.
* The indices are saved to `data/cache/patient_split.json`, so every script uses the *same* split.

The supplied 1,000-scan **test split is never used** for training or model selection.
""")
code("""
from src.preprocessing import patient_split, load_cache

show_code(patient_split)
""")
md(f"{COMPUTED} Load the saved split and check it.")
code("""
train_images, train_meta = load_cache("train")      # memory-mapped, no RAM copy
test_images, test_meta = load_cache("test")
train_idx, val_idx = patient_split(train_meta)      # reads the saved JSON file

train_patients = set(train_meta.patient_id.iloc[train_idx])
val_patients = set(train_meta.patient_id.iloc[val_idx])

split_table = pd.DataFrame({
    "train": train_meta.label.iloc[train_idx].value_counts().sort_index().values,
    "validation": train_meta.label.iloc[val_idx].value_counts().sort_index().values,
    "test": test_meta.label.value_counts().sort_index().values,
}, index=config.CLASS_NAMES)
split_table.loc["TOTAL scans"] = split_table.sum()
split_table.loc["patients"] = [len(train_patients), len(val_patients), test_meta.file_name.str.split("-").str[1].nunique()]
display(split_table)
print("Patients in BOTH train and validation:", len(train_patients & val_patients))
""")

# ---------------------------------------------------------------- 2. preprocessing
md("""
## 2. Preprocessing — and why padding was replaced

In Part 1 I padded a wide scan to 224 × 224 (`ImageOps.pad`). The data checks then showed a problem:
""")
code("""
sizes = train_meta.orig_width.astype(str) + "×" + train_meta.orig_height.astype(str)
size_table = pd.crosstab(sizes, train_meta.label.map(dict(enumerate(config.CLASS_NAMES))))
display(size_table)
print("Test split sizes:", dict((test_meta.orig_width.astype(str) + "×" + test_meta.orig_height.astype(str)).value_counts()))
""")
md("""
**What the table shows (🟦 computed above):** every 512 × 512 scan is **DME**, and 512 × 512 is
the only size that needs *no* black padding bars. A padded model can therefore learn the shortcut
"no bars → DME" instead of looking at the retina. The test split is mostly 1024 × 496, which
padding shrinks to a thin strip.

**Final preprocessing (identical in training, evaluation and the Streamlit app):**
1. convert to grayscale,
2. **resize directly to 224 × 224** (no padding bars for any size),
3. scale pixels to 0–1, copy the gray channel 3× (ImageNet models expect RGB),
4. normalise with the ImageNet mean / std.

Every checkpoint stores its `resize_mode`, so the app always uses the preprocessing the model was trained with.
Section 6b compares padding vs direct resize on the same validation split.
""")
code("""
from src.preprocessing import preprocess_pil, to_model_tensor

show_code(preprocess_pil)
show_code(to_model_tensor)
""")
md(f"{COMPUTED} The same wide training scan, padded vs directly resized.")
code("""
from datasets import load_dataset

wide_row = int(np.flatnonzero((train_meta.orig_width == 1536).to_numpy())[0])
original = load_dataset(config.HF_DATASET_ID, split="train")[wide_row]["image"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, img, title in zip(axes,
                          [original, preprocess_pil(original, mode="pad"), preprocess_pil(original, mode="stretch")],
                          ["Original", "Pad (Part 1 experiment)", "Direct resize (final)"]):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{title}\\n{np.asarray(img).shape[1]} × {np.asarray(img).shape[0]}")
    ax.axis("off")
plt.tight_layout()
plt.show()

model_input = to_model_tensor(preprocess_pil(original))
print("Model input shape:", tuple(model_input.shape), "| mean ≈", round(model_input.mean().item(), 3))
""")

# ---------------------------------------------------------------- 3. batches
md("""
## 3. Batch creation (Dataset → DataLoader → GPU batch)

In Part 1 I stacked **two** scans by hand into a `[2, 1, 224, 224]` batch. Training does the same
thing automatically for 64 scans at a time:

* `OCTArrayDataset.__getitem__` returns **one** cached 224 × 224 scan (uint8) and its label.
* The `DataLoader` stacks them into a batch. A **class-balanced sampler** draws rare classes
  (DRUSEN 8 %) as often as common ones (NORMAL 47 %).
* `prepare_batch` moves the batch to the GPU, applies **augmentation** (training only: random flip,
  small rotation/zoom/shift, brightness/contrast) and normalises it.
""")
code("""
from src.dataset import OCTArrayDataset, make_loaders, prepare_batch, augment_batch, get_device

show_code(OCTArrayDataset)
show_code(prepare_batch)
""")
md(f"{COMPUTED} Draw one real training batch of 8 scans.")
code("""
device = get_device()
train_loader, val_loader, test_loader = make_loaders(batch_size=8, samples_per_epoch=64, val_limit=4000)
images_uint8, labels = next(iter(train_loader))
batch = prepare_batch(images_uint8, device, augment=True)

print("Raw batch from DataLoader:", tuple(images_uint8.shape), images_uint8.dtype)
print("Model-ready batch        :", tuple(batch.shape), batch.dtype, "on", batch.device)
print("Labels                   :", [config.CLASS_NAMES[i] for i in labels.tolist()])

fig, axes = plt.subplots(1, 8, figsize=(18, 2.8))
for ax, img, lab in zip(axes, batch.cpu(), labels):
    ax.imshow(img[0], cmap="gray")
    ax.set_title(config.CLASS_NAMES[lab])
    ax.axis("off")
plt.suptitle("One augmented, class-balanced training batch")
plt.show()
""")

# ---------------------------------------------------------------- 4. model
md("""
## 4. Model creation — ResNet18 with transfer learning

`build_model` loads a network pre-trained on ImageNet (1.2 M everyday photos). Its early layers
already detect edges and textures, which also helps on OCT scans. We only **replace the last layer**:
ImageNet has 1,000 classes, we need 4 (CNV, DME, DRUSEN, NORMAL). A Dropout layer is added before it
for regularisation.

* `mode="finetune"` → all layers keep learning (used for the final models).
* `mode="feature_extract"` → the pretrained layers are frozen; only the new last layer learns.
""")
code("""
from src.models import build_model, count_parameters, load_checkpoint

show_code(build_model)
""")
md(f"{COMPUTED} Build ResNet18 and pass the batch through it.")
code("""
resnet18 = build_model("resnet18", mode="finetune").to(device)
total, trainable = count_parameters(resnet18)
print("New final layer:", resnet18.fc)
print(f"Parameters: {total:,} (trainable {trainable:,})")

with torch.no_grad():
    logits = resnet18.eval()(batch)
print("Output shape:", tuple(logits.shape), "→ one score per class for each of the 8 scans")
print("Untrained-head probabilities for scan 0:", torch.softmax(logits[0], 0).cpu().numpy().round(3))
""")

# ---------------------------------------------------------------- 5. training loop
md("""
## 5. The training loop

`run_epoch` is one pass over the data. For every batch:
1. **Forward pass:** the model produces scores (logits).
2. **Loss:** cross-entropy measures how wrong the scores are.
3. **Backward pass:** `loss.backward()` computes gradients (the chain rule, as in my NumPy example).
4. **Update:** `optimizer.step()` moves the weights a little against the gradient.

Settings used in `train()`:
* **Adam** optimiser, learning rate 1e-4 for pretrained models.
* `weight_decay=1e-4` = **L2 regularisation** (it penalises large weights).
* **ReduceLROnPlateau** lowers the learning rate when validation F1 stops improving.
* **Early stopping** ends training after 3 epochs without improvement.

In validation mode (`optimizer=None`) there are no gradients and no augmentation.
""")
code("""
from src.train import run_epoch, train

show_code(run_epoch)
""")
md("""
The part of `train()` that runs the epochs and **chooses the best checkpoint**:
""")
code("""
train_source = inspect.getsource(train)
start = train_source.index("    for epoch in range(start_epoch")
end = train_source.index("    if history[\\"status\\"] == \\"running\\"")
display(Markdown("```python\\n" + train_source[start:end] + "\\n```"))
""")
md("""
### 5b. ⚠️ OPTIONAL — retrain ResNet18 (switched OFF by default)

Retraining takes about **15 minutes** on the MacBook GPU and **overwrites** the saved checkpoint.
Leave `RETRAIN = False` for the viva: all later cells use the saved weights.
""")
code("""
RETRAIN = False   # ← change to True only if you really want to retrain (~15 min)

if RETRAIN:
    import subprocess
    command = [sys.executable, "-m", "src.train", "--model", "resnet18", "--mode", "finetune",
               "--epochs", "6", "--samples-per-epoch", "16000", "--val-limit", "4000",
               "--lr", "1e-4", "--batch-size", "64"]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
else:
    print("Skipped retraining - using the saved run in models/resnet18_finetune_*")
""")

# ---------------------------------------------------------------- 6. recorded epochs + best
md(f"""
## 6. The recorded epochs and best-checkpoint selection

{SAVED} During training, `src/train.py` wrote one row per epoch to
`models/resnet18_finetune_history.json`.

**Training setup (shortened run, clearly labelled):** each "epoch" = 16,000 class-balanced scans drawn
from the 97,477 training scans (a full pass would take ~9 min per epoch on this Mac). Validation =
fixed 3,862-scan stratified subset of the patient-level validation split.
""")
code("""
history = json.loads((PROJECT_ROOT / "models/resnet18_finetune_history.json").read_text())
epochs = pd.DataFrame(history["epochs"])[
    ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "train_seconds"]]
display(epochs.round(4).style.highlight_max(subset=["val_macro_f1"], color="#FFF3B0"))
print("Status:", history["status"], "| best epoch:", history["best_epoch"],
      "| best val macro-F1:", round(history["best_val_macro_f1"], 4))
""")
code("""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs.epoch, epochs.train_loss, "o-", label="train")
ax1.plot(epochs.epoch, epochs.val_loss, "o-", label="validation")
ax1.set_title("Loss (lower is better)"); ax1.set_xlabel("epoch"); ax1.legend()
ax2.plot(epochs.epoch, epochs.train_acc, "o-", label="train")
ax2.plot(epochs.epoch, epochs.val_acc, "o-", label="validation")
ax2.axvline(history["best_epoch"], color="#F4B400", ls="--", label="best checkpoint")
ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
plt.suptitle("🟨 Saved training history - ResNet18 fine-tune (direct resize)")
plt.tight_layout(); plt.show()
""")
md("""
**How the best checkpoint is chosen.** After every epoch the model is scored on the validation set.
If its **macro-F1** (the average F1 of the 4 classes, so the rare DRUSEN class counts as much as
NORMAL) is the best so far, the weights are saved to `models/resnet18_finetune_best.pt`.
Later epochs that score lower do **not** overwrite it. The test set plays no part in this choice.
""")
md(f"{SAVED} Reload the best checkpoint (weights + preprocessing settings).")
code("""
best_model, ckpt = load_checkpoint(PROJECT_ROOT / "models/resnet18_finetune_best.pt", device)
print("Run           :", ckpt["run"])
print("Architecture  :", ckpt["model_name"], "/", ckpt["mode"])
print("Saved at epoch:", ckpt["epoch"], "| val acc", round(ckpt["val_acc"], 4),
      "| val macro-F1", round(ckpt["val_macro_f1"], 4))
print("Class order   :", ckpt["class_names"])
print("Preprocessing :", ckpt["preprocessing"], "| resize_mode =", ckpt["resize_mode"])
""")
md(f"""
### 6b. Padding vs direct resize on the same validation split

{SAVED} Both ResNet18 runs used identical settings; only the resize step differed. Each was scored on
**all 10,832 validation scans** by `python -m src.compare_preprocessing`
(file `outputs/metrics/preprocessing_comparison.md`). Test numbers there are for information only.
""")
code("""
comparison_file = PROJECT_ROOT / "outputs/metrics/preprocessing_comparison.md"
if comparison_file.exists():
    display(Markdown(comparison_file.read_text()))
else:
    print("Comparison not run yet: python -m src.compare_preprocessing")
""")

# ---------------------------------------------------------------- 7. evaluation
md(f"""
## 7. Evaluation on the unseen test set

{COMPUTED} The best checkpoint predicts all 1,000 test scans (250 per class, 635 patients, none of
them seen in training). We report accuracy, precision, recall and F1 for each class, and a confusion
matrix (rows = true class, columns = predicted class).
""")
code("""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

predictions, truths = [], []
best_model.eval()
with torch.no_grad():
    for images_uint8, labels in test_loader:
        logits = best_model(prepare_batch(images_uint8, device, augment=False))
        predictions.append(logits.argmax(1).cpu())
        truths.append(labels)
predictions = torch.cat(predictions).numpy()
truths = torch.cat(truths).numpy()

print(f"Test accuracy: {accuracy_score(truths, predictions):.4f}  ({len(truths)} scans)\\n")
print(classification_report(truths, predictions, target_names=config.CLASS_NAMES, digits=3))
ConfusionMatrixDisplay(confusion_matrix(truths, predictions), display_labels=config.CLASS_NAMES).plot(cmap="YlOrBr")
plt.title("🟦 ResNet18 (direct resize) - test confusion matrix")
plt.show()
""")
md("""
**Reading the confusion matrix:** the diagonal is correct predictions. The largest off-diagonal
number shows which two classes the model mixes up most. Drusen and CNV are both signs of macular
degeneration and can look alike on OCT, so that pair is the one to watch.
The comparison of all five architectures is in `outputs/metrics/eval_model_comparison.md` and the
Streamlit **Model Performance** page.
""")

# ---------------------------------------------------------------- 8. prediction
md(f"""
## 8. One prediction, exactly as the app does it

{COMPUTED} An original test JPEG from `sample_images/` goes through `src/inference.py`, the same code
the Streamlit app uses: validate file → preprocess with the checkpoint's `resize_mode` → model → softmax.
""")
code("""
from src import inference

sample_path = sorted((PROJECT_ROOT / "sample_images").glob("DRUSEN_test_*.jpeg"))[0]
scan = inference.validate_upload(sample_path.read_bytes(), sample_path.name)
result = inference.predict_image(best_model, scan, device, ckpt["class_names"],
                                 resize_mode=ckpt["resize_mode"])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 3.8), gridspec_kw={"width_ratios": [2, 1, 1.3]})
ax1.imshow(scan.convert("L"), cmap="gray"); ax1.set_title(f"Uploaded scan ({scan.size[0]}×{scan.size[1]})"); ax1.axis("off")
ax2.imshow(result["preprocessed_uint8"], cmap="gray"); ax2.set_title("Model input 224×224"); ax2.axis("off")
ax3.barh(list(result["probabilities"]), [p * 100 for p in result["probabilities"].values()], color="#F4B400")
ax3.set_xlim(0, 100); ax3.set_xlabel("probability (%)"); ax3.set_title("Model output")
plt.tight_layout(); plt.show()

print("File (true class in name):", sample_path.name)
print(f"Predicted: {result['class_name']}  (confidence {result['confidence']:.1%})")
print("Educational model output - not a medical diagnosis.")
""")
md("""
### Summary for the viva
* **Split by patient** → no leakage; the test set is used only at the end.
* **Direct resize** instead of padding → removes the "no black bars = DME" shortcut.
* **Transfer learning** (ResNet18 + new 4-class layer) learns quickly from limited training time.
* **Best checkpoint chosen on validation macro-F1**, and reloaded here and in the app without retraining.
* The app (`streamlit run app.py`) uses exactly the same preprocessing and class order.
""")


def main() -> None:
    nb = nbf.read(NB_PATH, as_version=4)
    keep = []
    for cell in nb.cells:
        if cell.cell_type == "markdown" and MARKER in cell.source:
            break
        keep.append(cell)
    # drop trailing empty code cells left at the end of the original notebook
    while keep and keep[-1].cell_type == "code" and not keep[-1].source.strip():
        keep.pop()
    nb.cells = keep + cells
    nbf.write(nb, NB_PATH)
    print(f"kept {len(keep)} original cells, appended {len(cells)} walkthrough cells")


if __name__ == "__main__":
    main()
