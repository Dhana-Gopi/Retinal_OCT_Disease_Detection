"""Central configuration shared by training, evaluation, Grad-CAM and the app.

Keeping every constant here guarantees that the preprocessing used during
training is exactly the preprocessing used during evaluation and in the app.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Hugging Face dataset (already downloaded to the local cache).
HF_DATASET_ID = "zacharielegault/Kermany2017-OCT"

# Folders inside the project.
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # preprocessed 224x224 uint8 arrays
MODELS_DIR = PROJECT_ROOT / "models"    # checkpoints + training histories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Class order is fixed by the dataset's ClassLabel feature.
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = len(CLASS_NAMES)

# ---- Training / evaluation / app preprocessing (single source of truth) ----
# 1. convert to 8-bit grayscale ("L")
# 2. resize directly to IMAGE_SIZE x IMAGE_SIZE (RESIZE_MODE = "stretch")
# 3. scale to [0, 1], repeat the gray channel 3x (pretrained nets expect RGB)
# 4. normalise with ImageNet mean / std
#
# Why "stretch" and not "pad": the notebook experiment padded scans to a square.
# The data checks showed that all 512x512 training scans are DME (the only size
# that needs no padding bars), so "no black bars" became a DME shortcut, and the
# test split is 94 % 1024x496 scans that padding shrinks to a thin strip.
# A padded ResNet18 run confirmed it (val DME recall 94 % on 512x512 scans vs
# 72 % on other sizes). Direct resizing removes the bar cue for every size.
IMAGE_SIZE = 224
RESIZE_MODE = "stretch"   # "pad" = the earlier experiment (kept for reproducibility)
PAD_FILL = 0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Validation split: fraction of *patients* held out from the training split.
VAL_PATIENT_FRACTION = 0.10
SEED = 42

MODEL_NAMES = ["custom_cnn", "resnet18", "vgg16", "mobilenet_v2", "densenet121"]
