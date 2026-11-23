# Retinal Disease Detection Using Deep Learning on OCT Images

A GUVI project that classifies retinal **OCT (Optical Coherence Tomography)** scans into
four classes with PyTorch:

| Label | Class | What it means (simple words) |
|---|---|---|
| 0 | **CNV** – Choroidal Neovascularization | new, leaky blood vessels grow under the retina ("wet" AMD) |
| 1 | **DME** – Diabetic Macular Edema | fluid swelling in the centre of the retina caused by diabetes |
| 2 | **DRUSEN** | small yellow deposits under the retina (early AMD sign) |
| 3 | **NORMAL** | healthy retina |

The project covers: data exploration, preprocessing and augmentation, a small neural
network written in pure **NumPy**, a **custom CNN**, **transfer learning** (ResNet18, VGG16,
MobileNetV2, DenseNet121), regularisation experiments, evaluation on an untouched test set,
**Grad-CAM** explanations and a **Streamlit** web app.

> **Disclaimer:** this is an educational prototype. It is **not** a medical device and must
> not be used to diagnose or treat anyone.

---

## 1. Results (summary)

<!-- RESULTS PENDING: filled in from outputs/metrics/eval_model_comparison.csv -->

| Model | Mode | Test accuracy | Test macro-F1 | Params | Inference (img/s) |
|---|---|---|---|---|---|
| Custom CNN | from scratch | PENDING | PENDING | 1.2 M | PENDING |
| ResNet18 | fine-tune | PENDING | PENDING | 11.2 M | PENDING |
| ResNet18 | feature extract | PENDING | PENDING | 11.2 M | PENDING |
| MobileNetV2 | fine-tune | PENDING | PENDING | 2.2 M | PENDING |
| DenseNet121 | fine-tune | PENDING | PENDING | 7.0 M | PENDING |
| VGG16 | fine-tune | PENDING | PENDING | 134.3 M | PENDING |

Best model (used by the app): **PENDING** → `models/best_model.pt`.
Full details: [`reports/Project_Report.md`](reports/Project_Report.md).

---

## 2. Project structure

```
Retinal_OCT_Project/
├── OCT_dataset.ipynb          # main notebook: walks through every step with outputs
├── app.py                     # Streamlit web app (upload a scan -> prediction + Grad-CAM)
├── src/                       # reusable Python modules
│   ├── config.py              # all constants (class names, image size, paths, seed)
│   ├── preprocessing.py       # grayscale -> resize 224x224 cache + patient-level split
│   ├── data_checks.py         # dataset exploration figures / statistics
│   ├── numpy_nn.py            # tiny neural network in pure NumPy (fundamentals)
│   ├── dataset.py             # PyTorch Dataset/DataLoader, balanced sampler, augmentation
│   ├── models.py              # custom CNN + pretrained models, checkpoint save/load
│   ├── train.py               # training loop, early stopping, best-checkpoint saving
│   ├── evaluate.py            # test-set metrics, confusion matrices, model comparison
│   ├── gradcam.py             # Grad-CAM heat-maps
│   ├── efficiency.py          # speed / memory measurements
│   └── inference.py           # single-image prediction used by the app
├── scripts/
│   ├── run_training_queue.sh  # trains all models one after another
│   ├── make_report_docx.py    # reports/Project_Report.md -> .docx
│   └── make_presentation.py   # builds reports/Retinal_OCT_Presentation.pptx
├── models/                    # *_best.pt checkpoints + *_history.json training curves
├── outputs/
│   ├── figures/               # all plots (data_*, numpy_*, curves_*, eval_*, gradcam_*)
│   └── metrics/               # JSON / CSV metric files
├── reports/                   # Project_Report.md/.docx, presentation, viva Q&A
├── sample_images/             # a few example OCT scans to try in the app
├── data/                      # (not in Git) preprocessed cache, ~5 GB
├── requirements.txt
└── README.md
```

---

## 3. Setup

### 3.1 Create the Python environment (one time)

Open a terminal (in VS Code: menu **Terminal → New Terminal**) and run:

```bash
conda create -n Retinal_OCT_Project python=3.14 -y
conda activate Retinal_OCT_Project
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions this project was built and tested with
(PyTorch 2.15, torchvision 0.29, Streamlit 1.67, …). On a Mac with Apple Silicon PyTorch
uses the **MPS** GPU automatically; on other machines it uses CUDA if available, else CPU.

### 3.2 Download the dataset (one time, ~5 GB)

The data comes from Hugging Face: [`zacharielegault/Kermany2017-OCT`](https://huggingface.co/datasets/zacharielegault/Kermany2017-OCT)
(the OCT2017 / Kermany 2018 dataset). Download it into the local Hugging Face cache:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('zacharielegault/Kermany2017-OCT', repo_type='dataset')"
```

The dataset is **not** stored in this repository (too large for GitHub).

---

## 4. How to run each step

Always run commands from the **project root folder** (the folder that contains `app.py`)
with the environment activated.

| Step | Command | What it does / output |
|---|---|---|
| 1. Preprocess + cache | `python -m src.preprocessing` | builds `data/cache/` (224×224 scans, metadata, patient split). Run once (~10 min). |
| 2. Data exploration | `python -m src.data_checks` | class counts, image sizes, sample grid → `outputs/figures/data_*.png` |
| 3. NumPy fundamentals | `python -m src.numpy_nn` | perceptron/MLP with ReLU vs Sigmoid vs Tanh → `outputs/figures/numpy_*.png` |
| 4. Train all models | `bash scripts/run_training_queue.sh` | trains every model one after another → `models/*_best.pt`, `models/*_history.json`, `logs/` |
| 4b. Train one model | `python -m src.train --model resnet18 --mode finetune --epochs 6 --samples-per-epoch 16000 --val-limit 4000` | see `python -m src.train --help` for all options |
| 5. Evaluate on test set | `python -m src.evaluate` | metrics, confusion matrices, comparison table → `outputs/metrics/eval_*`, `outputs/figures/eval_*`, `curves_*` |
| 6. Grad-CAM | `python -m src.gradcam` | heat-maps → `outputs/figures/gradcam_*.png` |
| 7. Web app | `streamlit run app.py` | opens http://localhost:8501 |
| 8. Report / slides | `python scripts/make_report_docx.py` and `python scripts/make_presentation.py` | `reports/Project_Report.docx`, `reports/Retinal_OCT_Presentation.pptx` |

Other useful training examples (the same settings the queue uses):

```bash
# Custom CNN from scratch (main run)
python -m src.train --model custom_cnn --run-name custom_cnn_finetune --epochs 8 --samples-per-epoch 16000 --batch-size 64 --lr 1e-3 --dropout 0.4 --weight-decay 1e-4 --val-limit 4000

# Regularisation experiment: custom CNN without BatchNorm
python -m src.train --model custom_cnn --run-name cnn_no_batchnorm --no-bn --epochs 4 --samples-per-epoch 8000 --lr 1e-3 --val-limit 4000

# Transfer learning: frozen backbone (feature extraction)
python -m src.train --model resnet18 --mode feature_extract --epochs 4 --samples-per-epoch 16000 --lr 1e-3 --val-limit 4000

# Resume an interrupted run: repeat the same command and add --resume
```

Smoke-test runs (`models/smoke_*`) only check that the pipeline works; they are not results.

### Regenerating the VGG16 checkpoint

The VGG16 checkpoint is about **512 MB**, larger than GitHub's 100 MB file limit, so it is
listed in `.gitignore`. Its training history (`models/vgg16_finetune_history.json`) and test
metrics are kept. To recreate the weights, run the VGG16 line from
`scripts/run_training_queue.sh` (about 30–40 minutes on a MacBook Air M-series), e.g.:

```bash
python -m src.train --model vgg16 --run-name vgg16_finetune --epochs 4 --samples-per-epoch 10000 --lr 5e-5 --batch-size 32 --val-limit 4000
```

---

## 5. Step-by-step in VS Code (for beginners)

**A. Open the project**
1. Open VS Code → **File → Open Folder…** → choose `Retinal_OCT_Project` → **Open**.

**B. Select the correct Python interpreter**
1. Press **Cmd + Shift + P** (Windows: Ctrl + Shift + P).
2. Type **Python: Select Interpreter** and press Enter.
3. Choose the entry that shows **Retinal_OCT_Project** (path ends in
   `/opt/anaconda3/envs/Retinal_OCT_Project/bin/python`).

**C. Open and run the notebook**
1. In the left Explorer panel click **OCT_dataset.ipynb**.
2. Click **Select Kernel** (top-right) → **Python Environments…** → **Retinal_OCT_Project**.
3. Run a cell with **Shift + Enter**, or click **Run All** at the top.

**D. Run commands in the terminal**
1. Menu **Terminal → New Terminal**. A terminal opens at the bottom, already inside the
   project folder.
2. If the prompt does not start with `(Retinal_OCT_Project)`, type
   `conda activate Retinal_OCT_Project` and press Enter.
3. Type any command from section 4, e.g. `python -m src.evaluate`, and press Enter.

**E. Start the web app**
1. In the terminal type `streamlit run app.py` and press Enter.
2. Your browser opens automatically; if not, open **http://localhost:8501**.
3. Click **Browse files**, pick a scan from `sample_images/`, and read the prediction,
   confidence and Grad-CAM heat-map.
4. To stop the app, click inside the terminal and press **Ctrl + C**.

---

## 6. Key design decisions (short version)

- **Patient-level split** – the same patient never appears in both training and
  validation (prevents "data leakage" that would make results look too good).
  Train 97,477 scans (4,312 patients) / validation 10,832 scans (473 patients). The official 1,000-scan
  test set (250 per class) shares **no patients** with training and is used only once,
  for the final evaluation.
- **Direct resize to 224×224 (no padding).** An earlier padded version created a shortcut:
  all 512×512 training scans are DME and were the only ones without black bars, so the
  model could learn "no bars = DME". Padding also shrank the 1024×496 test scans to a thin
  strip. See the report, section 3.2 ("lesson learned").
- **Grayscale → 3 channels + ImageNet normalisation** so pretrained ImageNet weights work.
- **Class-balanced sampling** – NORMAL is 47 % of training scans but DRUSEN only 8 %.
- **On-GPU augmentation** – flips, small rotations/zoom/shift, brightness/contrast.
- **Shortened training** – measured speed on a MacBook Air (e.g. VGG16 ≈ 29 img/s), so
  each "epoch" uses 16,000 balanced scans, ≤ 6 epochs, early stopping (patience 3).
- **Best checkpoint chosen by validation macro-F1**, never by test results.

---

## 7. Limitations

- Shortened training (sampled epochs) because of laptop compute; longer training could
  improve results.
- One dataset from one source; the model may not generalise to other OCT machines.
- Test images are mostly 1024×496 (0.4 % of training scans) – a distribution shift; all
  512×512 training scans are DME (dataset bias).
- Single label per scan; real eyes can have more than one condition.
- Grad-CAM shows *where* the model looks, not *why*; it is not a clinical explanation.

## 8. Disclaimer

This software is for **learning purposes only**. It is not a medical device, has not
been clinically validated and must not be used for diagnosis or treatment decisions.
Always consult a qualified ophthalmologist.

## 9. Dataset credit

Kermany D., Zhang K., Goldbaum M. (2018). *Labeled Optical Coherence Tomography (OCT) and
Chest X-Ray Images for Classification.* Mendeley Data, v2. Hugging Face mirror:
`zacharielegault/Kermany2017-OCT`.
