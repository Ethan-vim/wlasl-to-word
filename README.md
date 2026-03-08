# Live American Sign Language Recognition using WLASL

A real-time ASL word-level recognition system that captures live webcam video, processes hand/body movements frame-by-frame using MediaPipe, and predicts the signed word using deep learning models trained on the [WLASL dataset](https://github.com/dxli94/WLASL).

## Table of Contents

- **[Architecture](#architecture)** — Line 61
- **[Quick Start](#quick-start)** — Line 93
  - [1. Environment Setup](#1-environment-setup) — Line 95
    - [Installing PyTorch with CUDA support](#installing-pytorch-with-cuda-support) — Line 133
  - [2. Download the Dataset](#2-download-the-dataset) — Line 169
    - [Option A: Kaggle (Recommended)](#option-a-kaggle-recommended--fastest) — Line 171
    - [Option B: Official WLASL scripts](#option-b-official-wlasl-scripts-url-based) — Line 236
    - [Validate downloaded videos](#validate-downloaded-videos) — Line 247
    - [End-to-end quick start with Kaggle](#end-to-end-quick-start-with-kaggle) — Line 264
    - [Choosing a wlasl_variant with Kaggle](#choosing-a-wlasl_variant-with-kaggle) — Line 305
    - [Device-specific configuration after Kaggle download](#device-specific-configuration-after-kaggle-download) — Line 320
  - [3. Preprocess Data](#3-preprocess-data) — Line 383
    - [Working with multiple variants](#working-with-multiple-variants) — Line 409
  - [4. Train a Model](#4-train-a-model) — Line 464
  - [5. Evaluate](#5-evaluate) — Line 499
  - [6. Run the Live Demo](#6-run-the-live-demo) — Line 525
  - [7. Single Video Prediction](#7-single-video-prediction) — Line 555
  - [8. Export to ONNX](#8-export-to-onnx) — Line 607
  - [9. Run Tests](#9-run-tests) — Line 635
- **[Project Structure](#project-structure)** — Line 678
- **[Configuration Guide](#configuration-guide)** — Line 745
  - [Auto-Configure for Your Hardware](#auto-configure-for-your-hardware) — Line 759
- **[Approach Details](#approach-details)** — Line 881
  - [Approach A: Pose/Keypoint Transformer](#approach-a-posekeypoint-transformer) — Line 883
  - [Approach B: RGB Video Classifier](#approach-b-rgb-video-classifier) — Line 903
  - [Approach C: Hybrid Fusion](#approach-c-hybrid-fusion) — Line 911
- **[Troubleshooting](#troubleshooting)** — Line 919
  - [HTML files masquerading as videos](#html-files-masquerading-as-videos) — Line 921
  - [MediaPipe installation issues](#mediapipe-installation-issues) — Line 924
  - [CUDA out of memory](#cuda-out-of-memory) — Line 936
  - [Webcam not detected](#webcam-not-detected) — Line 942
  - [Low accuracy](#low-accuracy) — Line 948
  - [Diagnosing partial data](#diagnosing-partial-data-most-common-issue) — Line 953
  - [wlasl_variant / num_classes mismatch](#wlasl_variant--num_classes-mismatch) — Line 1000
- **[Recommended Configurations](#recommended-configurations)** — Line 1006
  - [WLASL100 (recommended starting point)](#wlasl100-recommended-starting-point) — Line 1010
  - [WLASL300](#wlasl300) — Line 1043
  - [WLASL1000 / WLASL2000](#wlasl1000--wlasl2000) — Line 1065
  - [Video Classifier (Approach B)](#video-classifier-approach-b) — Line 1089
  - [Fusion (Approach C)](#fusion-approach-c) — Line 1105
- **[Tips & Best Practices](#tips--best-practices)** — Line 1124
  - [Hardware-Specific Setup](#hardware-specific-setup) — Line 1126
  - [Training with Limited Data](#training-with-limited-data) — Line 1149
  - [Improving Accuracy](#improving-accuracy) — Line 1161
  - [What to Expect](#what-to-expect) — Line 1172
  - [Common Pitfalls](#common-pitfalls) — Line 1184
- **[Recommended Library & CUDA Versions](#recommended-library--cuda-versions)** — Line 1195
  - [PyTorch ↔ CUDA Compatibility](#pytorch--cuda-compatibility) — Line 1199
  - [MediaPipe](#mediapipe) — Line 1218
  - [Other Key Libraries](#other-key-libraries) — Line 1231
- **[Citation](#citation)** — Line 1248
- **[License](#license)** — Line 1260

---

## Architecture

```
                        APPROACH A (Pose/Keypoint)
                    +-------------------------------+
                    |  MediaPipe     Transformer/   |
Webcam  --> Frame --+  Holistic  --> LSTM Encoder --+--> Predicted
Feed        Buffer  |  Keypoints    (T, 543*6)     |    Gloss +
(OpenCV)    (T=64)  |  + Velocity                  |    Confidence
                    +-------------------------------+
                    |                               |
                    |  APPROACH B (RGB Video)        |
                    |  R(2+1)D / SlowFast / R3D    |
                    |  (B, 3, T, 224, 224)         |
                    +-------------------------------+
                    |                               |
                    |  APPROACH C (Hybrid Fusion)    |
                    |  Concat / Cross-Attention     |
                    |  of Pose + Video features     |
                    +-------------------------------+
```

**Three approaches are implemented:**

| Approach | Model | Input | WLASL-100 Top-1 (expected) |
|----------|-------|-------|----------------------------|
| A - Pose Transformer | Transformer Encoder / BiLSTM | MediaPipe keypoints (T, 543, 6) with velocity | 60–70% |
| B - Video Classifier | R(2+1)D-18, R3D-18, SlowFast | RGB frames (3, T, 224, 224) | 65–75% |
| C - Hybrid Fusion | Concat / Cross-Attention fusion of A+B | Both streams | 70–78% |

---

## Quick Start

### 1. Environment Setup

**Linux/macOS:**

```bash
git clone <this-repo-url>
cd "Live American Sign Language Recognition using WLASL"

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
git clone <this-repo-url>
cd "Live American Sign Language Recognition using WLASL"

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Windows (Command Prompt):**

```cmd
git clone <this-repo-url>
cd "Live American Sign Language Recognition using WLASL"

python -m venv .venv
.venv\Scripts\activate.bat

pip install -r requirements.txt
```

#### Installing PyTorch with CUDA support

The default `pip install -r requirements.txt` installs the CPU-only version of PyTorch. If you have an NVIDIA GPU, install the CUDA-enabled version **before** running `pip install -r requirements.txt` (or after, to overwrite the CPU version):

```bash
# CUDA 12.4 (recommended for modern GPUs — RTX 30xx/40xx/50xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (for older GPUs or driver versions)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only (default — no flag needed, but explicit if you want to be sure)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

To check which CUDA version your driver supports:

```bash
nvidia-smi    # look for "CUDA Version" in the top-right corner
```

After installing, verify CUDA is available:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, version: {torch.version.cuda}')"
```

These commands work identically on Linux, macOS, and Windows.

> **Windows note:** All `python` commands in this README work on both platforms. When you see `\` at the end of a line (bash line continuation), replace it with `` ` `` (backtick) in PowerShell or `^` in Command Prompt. Platform-specific shell commands (file operations, venv activation) show both variants where they differ.

---

### 2. Download the Dataset

#### Option A: Kaggle (Recommended — fastest)

The full WLASL video archive (~12,000 videos, ~5 GB) is available on [Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed). This is the fastest way to get the data since it downloads as a single archive.

**One-time Kaggle API setup:**

**Linux/macOS:**

```bash
# kaggle is already included in requirements.txt, so if you ran
# pip install -r requirements.txt, it's already installed.
# Otherwise: pip install kaggle

# Get your API token from https://www.kaggle.com/settings → "Create New Token"
# Move the downloaded kaggle.json to ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Windows (PowerShell):**

```powershell
# Get your API token from https://www.kaggle.com/settings → "Create New Token"
# Move the downloaded kaggle.json to %USERPROFILE%\.kaggle\
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
# No chmod needed on Windows
```

**Windows (Command Prompt):**

```cmd
mkdir "%USERPROFILE%\.kaggle"
move "%USERPROFILE%\Downloads\kaggle.json" "%USERPROFILE%\.kaggle\kaggle.json"
```

**Download:**

```bash
python scripts/download_kaggle.py --subset WLASL100
```

This downloads **all** ~12K videos to `data/raw/` (the full archive, regardless of `--subset`), fetches the annotation JSON, and prints a summary of how many videos match the chosen subset. The `--subset` flag only controls the summary output — the download itself always fetches the full dataset.

You can also use the Kaggle CLI directly:

**Linux/macOS:**

```bash
kaggle datasets download -d risangbaskoro/wlasl-processed -p data/_kaggle_download --unzip
mv data/_kaggle_download/videos/*.mp4 data/raw/
rm -rf data/_kaggle_download
```

**Windows (PowerShell):**

```powershell
kaggle datasets download -d risangbaskoro/wlasl-processed -p data\_kaggle_download --unzip
Move-Item data\_kaggle_download\videos\*.mp4 data\raw\
Remove-Item -Recurse -Force data\_kaggle_download
```

> **Note:** The Kaggle archive contains all ~12K videos for all WLASL variants (100–2000). You only download once — the preprocessing step (Step 3) filters to your chosen subset.

#### Option B: Official WLASL scripts (URL-based)

```bash
# Download the WLASL annotation JSON and print video download instructions
python scripts/download_wlasl.py --subset WLASL100
```

This creates `data/annotations/WLASL_v0.3.json` and the directory structure under `data/`. Follow the printed instructions to download the actual video files from the [official WLASL repo](https://github.com/dxli94/WLASL) or a community mirror, then place them in `data/raw/`.

> **Note:** Many original WLASL URLs have expired. The Kaggle option (Option A) typically provides significantly more videos.

#### Validate downloaded videos

Many WLASL URLs have expired. When a URL is dead, servers often return an HTML redirect page (saved as `.mp4`) instead of a 404. This applies to both Kaggle and URL-based downloads. Run the validator before preprocessing to remove these fake files:

```bash
# Report how many invalid files exist
python scripts/validate_videos.py --video-dir data/raw

# Delete invalid files (HTML redirects and empty files)
python scripts/validate_videos.py --video-dir data/raw --delete

# Delete and save a list of valid video IDs for reference
python scripts/validate_videos.py --video-dir data/raw --delete --save-valid data/valid_ids.txt
```

The validator checks the first 256 bytes of each file for HTML signatures (`<!DOCTYPE html>`, `<html>`, etc.) and reports counts of valid / HTML / empty files. The preprocessing pipeline also skips unreadable files automatically, but cleaning them up first saves processing time.

#### End-to-end quick start with Kaggle

If you want the fastest path from zero to training, run these commands in order (all `python` commands are cross-platform):

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Configure Kaggle API (one-time — see "One-time Kaggle API setup" above)

# 3. Download all videos from Kaggle (~5 GB, ~12K videos)
#    --subset only controls the annotation summary printed after download;
#    the full archive is always downloaded regardless of the variant chosen.
python scripts/download_kaggle.py --subset WLASL100

# 4. Validate and clean up bad files
python scripts/validate_videos.py --video-dir data/raw --delete

# 5. Extract keypoints
python -m src.data.preprocess --data-dir data --subset WLASL100 --mode keypoints

# 6. Train (see device-specific configs below)
python -m src.training.train --config configs/pose_transformer.yaml

# 7. Evaluate
#    Linux/macOS uses \ for line continuation; Windows PowerShell uses `
python -m src.training.evaluate \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt \
    --split val --output-dir eval_results
```

On Windows PowerShell, the evaluate command (step 7) becomes:

```powershell
python -m src.training.evaluate `
    --config configs/pose_transformer.yaml `
    --checkpoint checkpoints/best_model.pt `
    --split val --output-dir eval_results
```

#### Choosing a `wlasl_variant` with Kaggle

The Kaggle archive contains all ~12K videos covering every WLASL variant. After downloading, you choose which variant to train on by setting `wlasl_variant` in your YAML config (and matching the `--subset` flag during preprocessing). The variant controls the number of sign classes — `num_classes` is auto-derived, so you never set it manually.

| Variant | Classes | Approx. Training Samples | Difficulty | Recommended For |
|---------|---------|--------------------------|------------|-----------------|
| `wlasl_variant: 100` | 100 | 800–1,200 | Easiest | First-time setup, prototyping, CPU training |
| `wlasl_variant: 300` | 300 | 2,000–3,500 | Moderate | Better vocabulary coverage with a GPU |
| `wlasl_variant: 1000` | 1,000 | 5,000–8,000 | Hard | Research, large-GPU setups |
| `wlasl_variant: 2000` | 2,000 | 8,000–12,000 | Hardest | Full dataset, consider fusion (Approach C) |

**Start with `wlasl_variant: 100`** — it has the most samples per class, trains fastest, and gives the highest per-class accuracy. Scale up once your pipeline is working.

Larger variants need more model capacity. See the [Recommended Configurations](#recommended-configurations) section for variant-specific hyperparameters (deeper layers, wider `d_model`, adjusted LR/dropout).

#### Device-specific configuration after Kaggle download

After downloading the Kaggle dataset, adjust `configs/pose_transformer.yaml` for your hardware before training (step 6). Set `wlasl_variant` to match the subset you preprocessed in step 5.

**CPU-only (no GPU):**

```yaml
approach: pose_transformer
wlasl_variant: 100          # match your preprocessed subset (100, 300, 1000, or 2000)
fp16: false                  # FP16 only works on CUDA
batch_size: 8                # smaller batches to avoid memory pressure
num_workers: 2
T: 64
d_model: 256
dropout: 0.3
lr: 1.0e-4
scheduler: onecycle
weighted_sampling: true
epochs: 100
```

Use `--device cpu` for inference and live demo. Stick to Approach A (pose_transformer) — video models are too slow on CPU for training.

**GPU / CUDA:**

```yaml
approach: pose_transformer
wlasl_variant: 100          # match your preprocessed subset (100, 300, 1000, or 2000)
fp16: true                   # faster training, lower memory
batch_size: 32               # increase to 64 for large GPUs
num_workers: 4
T: 64
d_model: 256
dropout: 0.3
lr: 1.0e-4
scheduler: onecycle
weighted_sampling: true
epochs: 100
```

Monitor GPU memory with `nvidia-smi`. If you run out of memory, reduce `batch_size` first, then `T`.

**Apple Silicon (M1/M2/M3/M4):**

```yaml
approach: pose_transformer
wlasl_variant: 100          # match your preprocessed subset (100, 300, 1000, or 2000)
fp16: false                  # MPS does not support FP16 reliably
batch_size: 16
num_workers: 2
T: 64
d_model: 256
dropout: 0.3
lr: 1.0e-4
scheduler: onecycle
weighted_sampling: true
epochs: 100
```

Use `--device cpu` for the live demo to avoid MPS overhead. Install MediaPipe with `pip install mediapipe-silicon` if the standard package fails.

---

### 3. Preprocess Data

The preprocessing pipeline is **source-agnostic** — it reads mp4 files from `data/raw/` regardless of whether they were downloaded via Kaggle (Option A) or URL-based scripts (Option B). No extra flags or options are needed.

Extract MediaPipe Holistic keypoints (543 landmarks per frame) from all valid videos:

```bash
python -m src.data.preprocess --data-dir data --subset WLASL100 --mode keypoints
```

This creates:
```
data/processed/<video_id>.npy       # keypoints, shape (T, 543, 3) — shared across variants
data/splits/WLASL100/train.csv      # variant-specific split files
data/splits/WLASL100/val.csv
data/splits/WLASL100/test.csv
```

To extract raw frames instead (needed for Approach B/C training):

```bash
python -m src.data.preprocess --data-dir data --subset WLASL100 --mode frames
```

Use `--max-workers N` to control parallel extraction (default: 4).

#### Working with multiple variants

WLASL100 ⊂ WLASL300 ⊂ WLASL1000 ⊂ WLASL2000 — each larger variant is a superset of the smaller ones. You can preprocess multiple variants without conflicts:

- **`data/raw/`** and **`data/processed/`** are shared — keypoints are stored by `video_id` and reused across variants. Already-extracted `.npy` files are skipped automatically.
- **`data/splits/WLASL{N}/`** is variant-specific — each variant gets its own `train/val/test.csv` files that never overwrite each other.

**With Kaggle (recommended):** Since Kaggle downloads all ~12K videos at once, you already have all the data. Just preprocess each variant:

```bash
# Download once (all variants included)
python scripts/download_kaggle.py --subset WLASL100

# Preprocess WLASL100
python -m src.data.preprocess --data-dir data --subset WLASL100

# Scale up — only new videos are extracted; WLASL100 splits are untouched
python -m src.data.preprocess --data-dir data --subset WLASL300
```

**With URL-based download:** Download annotations per variant, then add videos:

```bash
# First variant
python scripts/download_wlasl.py --subset WLASL100
# ...download videos to data/raw/...
python -m src.data.preprocess --data-dir data --subset WLASL100

# Scale up — only new videos are extracted; WLASL100 splits are untouched
python scripts/download_wlasl.py --subset WLASL300
# ...download the additional WLASL300 videos to data/raw/...
python -m src.data.preprocess --data-dir data --subset WLASL300
```

After two variants, your `data/` tree looks like:

```
data/
├── raw/                          # All videos (shared)
├── processed/                    # .npy keypoints by video_id (shared)
├── annotations/
│   └── WLASL_v0.3.json
└── splits/
    ├── WLASL100/                 # 100-class splits
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── WLASL300/                 # 300-class splits — coexist safely
        ├── train.csv
        ├── val.csv
        └── test.csv
```

---

### 4. Train a Model

Set `wlasl_variant` in your config to match the subset you preprocessed. All scripts automatically resolve split files from `data/splits/WLASL{N}/`.

```bash
# Approach A: Pose Transformer (recommended starting point)
python -m src.training.train --config configs/pose_transformer.yaml

# Approach B: Video Classifier
python -m src.training.train --config configs/video_classifier.yaml

# Approach C: Hybrid Fusion
python -m src.training.train --config configs/fusion.yaml
```

Training checkpoints are saved to `checkpoints/` and logs to `logs/`.

```bash
# Monitor training in real time
tensorboard --logdir logs/
```

To train on a different variant, either edit `wlasl_variant` in an existing config or use a separate config file:

```bash
# Copy and modify — only change wlasl_variant (num_classes is auto-derived)
cp configs/pose_transformer.yaml configs/pose_wlasl300.yaml          # Linux/macOS
# copy configs\pose_transformer.yaml configs\pose_wlasl300.yaml      # Windows

# Edit wlasl_variant: 300 in the new file
python -m src.training.train --config configs/pose_wlasl300.yaml
```

---

### 5. Evaluate

**Linux/macOS:**

```bash
python -m src.training.evaluate \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt \
    --split val \
    --output-dir eval_results
```

**Windows (PowerShell):**

```powershell
python -m src.training.evaluate `
    --config configs/pose_transformer.yaml `
    --checkpoint checkpoints/best_model.pt `
    --split val `
    --output-dir eval_results
```

This prints top-1/top-5 accuracy, per-class breakdown, and saves a confusion matrix heatmap to `eval_results/`.

---

### 6. Run the Live Demo

**Linux/macOS:**

```bash
python -m src.inference.live_demo \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt \
    --camera 0 \
    --device cpu
```

**Windows (PowerShell):**

```powershell
python -m src.inference.live_demo `
    --config configs/pose_transformer.yaml `
    --checkpoint checkpoints/best_model.pt `
    --camera 0 `
    --device cpu
```

**Controls:**
- `q` — quit
- `s` — save the current prediction to a log file

The demo runs three threads: a capture thread reads webcam frames continuously, an inference thread runs the model every 0.5 s on a rolling buffer of T frames, and the main thread renders the overlay. Predictions are smoothed over the last 5 inference windows and only displayed when confidence exceeds the configured threshold (default: 0.6).

---

### 7. Single Video Prediction

**Linux/macOS:**

```bash
# From a video file
python -m src.inference.predict \
    --video path/to/video.mp4 \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt

# From a pre-extracted keypoint .npy file
python -m src.inference.predict \
    --keypoints data/processed/12345.npy \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt

# Specify device (auto, cpu, cuda, mps)
python -m src.inference.predict \
    --video path/to/video.mp4 \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt \
    --device cpu
```

**Windows (PowerShell):**

```powershell
# From a video file
python -m src.inference.predict `
    --video path\to\video.mp4 `
    --config configs\pose_transformer.yaml `
    --checkpoint checkpoints\best_model.pt

# From a pre-extracted keypoint .npy file
python -m src.inference.predict `
    --keypoints data\processed\12345.npy `
    --config configs\pose_transformer.yaml `
    --checkpoint checkpoints\best_model.pt

# Specify device (auto, cpu, cuda, mps)
python -m src.inference.predict `
    --video path\to\video.mp4 `
    --config configs\pose_transformer.yaml `
    --checkpoint checkpoints\best_model.pt `
    --device cpu
```

Returns the predicted gloss, confidence score, and top-5 alternatives.

---

### 8. Export to ONNX

**Linux/macOS:**

```bash
python -m src.inference.export_onnx \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt \
    --output model.onnx \
    --verify \
    --benchmark
```

**Windows (PowerShell):**

```powershell
python -m src.inference.export_onnx `
    --config configs/pose_transformer.yaml `
    --checkpoint checkpoints/best_model.pt `
    --output model.onnx `
    --verify `
    --benchmark
```

`--verify` runs a forward pass through ONNX Runtime to confirm output shapes match. `--benchmark` measures average inference latency over 100 runs. Use `--opset N` to set the ONNX opset version (default: 17).

---

### 9. Run Tests

**Linux/macOS:**

```bash
source .venv/bin/activate
python -m pytest                          # full test suite (277 tests)
python -m pytest tests/test_augment.py    # specific test file
python -m pytest tests/test_dependencies.py  # dependency compatibility tests
python -m pytest -q                       # quiet output
```

Or without activating the venv:

```bash
.venv/bin/python -m pytest
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
python -m pytest                          # full test suite (277 tests)
python -m pytest tests\test_augment.py    # specific test file
python -m pytest -q                       # quiet output
```

Or without activating the venv:

```powershell
.venv\Scripts\python -m pytest
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
python -m pytest
```

Tests are fully isolated — they use pytest's `tmp_path` fixture for all file I/O and never touch project data, configs, or checkpoints. The `pyproject.toml` configures test discovery, verbose output, and warning suppression.

**Note:** `pytest` must be installed in the venv (`pip install pytest`). It is not listed in `requirements.txt` because it is a dev-only dependency.

The `test_dependencies.py` file (110 tests) verifies that every third-party library used by the `src/` code is importable, meets the minimum version from `requirements.txt`, and that the specific features relied upon (e.g. `batch_first` Transformers, `label_smoothing` in CrossEntropyLoss, seaborn heatmaps, ONNX Runtime sessions) work correctly with the installed versions.

---

## Project Structure

```
.
├── configs/
│   ├── pose_transformer.yaml    # Approach A defaults
│   ├── video_classifier.yaml    # Approach B defaults
│   └── fusion.yaml              # Approach C defaults
├── data/
│   ├── raw/                     # Downloaded video files — Kaggle or URL-based (shared)
│   ├── processed/               # Extracted keypoints as .npy (shared across variants)
│   ├── annotations/             # WLASL JSON annotation file
│   └── splits/
│       ├── WLASL100/            # train/val/test CSVs for 100-class variant
│       ├── WLASL300/            # train/val/test CSVs for 300-class variant
│       └── ...                  # one subdirectory per variant
├── src/
│   ├── data/
│   │   ├── preprocess.py        # Download, parse, extract & normalize keypoints
│   │   ├── augment.py           # Temporal & spatial augmentations
│   │   └── dataset.py           # PyTorch Dataset + motion feature computation
│   ├── models/
│   │   ├── pose_transformer.py  # Transformer & BiLSTM (Approach A)
│   │   ├── video_i3d.py         # 3D CNN backbones (Approach B)
│   │   └── fusion.py            # Multi-modal fusion (Approach C)
│   ├── training/
│   │   ├── config.py            # Config dataclass + YAML serialization
│   │   ├── train.py             # Training loop with mixup regularization
│   │   └── evaluate.py          # Metrics, TTA, confusion matrix
│   └── inference/
│       ├── predict.py           # Single-video prediction
│       ├── export_onnx.py       # ONNX export & latency benchmark
│       └── live_demo.py         # Real-time webcam demo
├── tests/
│   ├── conftest.py              # Shared fixtures (tmp datasets, keypoint helpers)
│   ├── test_augment.py          # Augmentation classes & pipeline presets
│   ├── test_config.py           # Config defaults, load/save, YAML roundtrip
│   ├── test_dataset.py          # Dataset, DataLoader, pad/crop, motion features
│   ├── test_evaluate.py         # Metrics, TTA flip, hard negatives, latency
│   ├── test_export_onnx.py      # ONNX export & verification
│   ├── test_live_demo.py        # FrameBuffer, prediction smoothing
│   ├── test_models.py           # PoseTransformer, PoseBiLSTM, FusionModel
│   ├── test_predict.py          # SignPredictor inference paths
│   ├── test_preprocess.py       # Normalization, annotation parsing, splits
│   ├── test_train.py            # Accuracy, mixup helpers
│   └── test_dependencies.py    # Library version & feature compatibility (110 tests)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_keypoint_visualization.ipynb
│   └── 03_error_analysis.ipynb
├── scripts/
│   ├── download_wlasl.py        # Download annotations, print video instructions
│   ├── download_kaggle.py       # Download videos from Kaggle (fast alternative)
│   ├── validate_videos.py       # Detect and remove HTML-disguised video files
│   ├── reset_configs.py         # Reset all configs/ to README.md recommended defaults
│   ├── check_mediapipe.py       # Verify MediaPipe installation and diagnose issues
│   └── auto_config.py           # Auto-detect hardware and generate optimized configs
├── checkpoints/                 # Saved model weights
├── logs/                        # TensorBoard training logs
├── pyproject.toml               # Pytest configuration
├── requirements.txt
├── CONTRIBUTING.md              # Contribution guide, commit standards, signing setup
└── STRUCTURE.md                 # Full workflow & dependency graph (which file calls which)
```

For a detailed breakdown of the entire pipeline — data flow diagrams, file dependency graphs, model architecture flow, and all CLI entry points — see [`STRUCTURE.md`](STRUCTURE.md).

---

## Configuration Guide

All hyperparameters live in YAML files under `configs/`. The table below shows **all** settings and their defaults from the `Config` dataclass (`src/training/config.py`). You only need to override values that differ from the defaults.

To reset all config files back to the recommended defaults:

```bash
python scripts/reset_configs.py                # reset all three configs
python scripts/reset_configs.py --only pose    # reset only pose_transformer.yaml
python scripts/reset_configs.py --only video   # reset only video_classifier.yaml
python scripts/reset_configs.py --only fusion  # reset only fusion.yaml
python scripts/reset_configs.py --dry-run      # preview without writing
```

### Auto-Configure for Your Hardware

Detect your GPU/CPU and generate an optimized config automatically. The script probes CUDA VRAM, Apple Silicon MPS, or CPU, classifies your hardware into a performance tier, and writes a ready-to-train config:

```bash
# Pose approach (recommended starting point)
python scripts/auto_config.py --approach pose

# Video approach
python scripts/auto_config.py --approach video --variant 100

# Fusion approach
python scripts/auto_config.py --approach fusion --variant 300

# Preview without writing
python scripts/auto_config.py --approach pose --dry-run

# Force CPU mode (e.g. no GPU or Apple Silicon)
python scripts/auto_config.py --approach pose --device cpu

# Back up existing config before overwriting
python scripts/auto_config.py --approach pose --backup
```

**Hardware tiers** (auto-detected from CUDA VRAM):

| Tier | VRAM | Pose `batch_size` | Video `batch_size` | `fp16` |
|------|------|-------------------|--------------------|--------|
| **high** | >= 16 GB | 64 | 16 | true |
| **mid** | >= 8 GB | 32 | 8 | true |
| **low** | >= 4 GB | 16 | 4 | true |
| **cpu** | MPS / CPU | 8 | 4 | false |

Windows (PowerShell / Command Prompt): the commands are identical — just run them in your terminal.

**Paths:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_dir` | Root data directory | `data` |
| `output_dir` | Output directory | `outputs` |
| `checkpoint_dir` | Checkpoint save directory | `checkpoints` |
| `log_dir` | Training log directory | `logs` |

**Dataset:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `wlasl_variant` | Dataset size: `100`, `300`, `1000`, `2000` | `100` |
| `num_classes` | Auto-derived from `wlasl_variant` — do not set manually | `100` |
| `T` | Temporal sequence length in frames | `64` |
| `image_size` | Spatial resolution for video models (Approach B/C) | `224` |
| `num_workers` | DataLoader worker processes | `4` |

**Model:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `approach` | `pose_transformer`, `pose_bilstm`, `video`, `fusion` | `pose_transformer` |
| `backbone` | Video backbone: `r2plus1d_18`, `r3d_18`, `mc3_18`, `slow_r50`, `slowfast_r50`, `x3d_m` | `r2plus1d_18` |
| `pretrained` | Use pretrained backbone weights (Approach B/C) | `true` |
| `num_keypoints` | Number of MediaPipe landmarks per frame (33 pose + 21 left hand + 21 right hand + 468 face) | `543` |
| `d_model` | Transformer/LSTM embedding dimension | `256` |
| `nhead` | Number of attention heads | `8` |
| `num_layers` | Number of encoder layers | `4` |
| `dropout` | Dropout rate | `0.3` |
| `use_motion` | Concatenate velocity (frame differences) with position features | `true` |
| `fusion` | Fusion strategy: `concat` or `attention` (Approach C only) | `concat` |
| `fusion_dim` | Fusion layer dimension (Approach C only) | `256` |

**Training:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epochs` | Maximum training epochs | `100` |
| `batch_size` | Training batch size | `32` |
| `lr` | Learning rate | `1e-4` |
| `weight_decay` | AdamW weight decay | `1e-4` |
| `warmup_epochs` | Linear warmup epochs before scheduler takes over | `10` |
| `label_smoothing` | Label smoothing factor (0 = disabled) | `0.1` |
| `grad_clip` | Max gradient norm for clipping | `1.0` |
| `fp16` | Mixed-precision (FP16) training | `true` |
| `weighted_sampling` | Weighted sampler to counter class imbalance | `false` |
| `early_stopping_patience` | Epochs without val improvement before stopping | `20` |
| `mixup_alpha` | Mixup interpolation parameter (0 = disabled) | `0.2` |
| `scheduler` | LR scheduler: `onecycle` or `cosine` (warmup + cosine annealing) | `onecycle` |

**Evaluation:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_tta` | Test-time augmentation via horizontal flip averaging | `false` |

**Inference / Live Demo:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `confidence_threshold` | Minimum confidence for live display | `0.6` |
| `smoothing_window` | Number of inference windows to smooth predictions over | `5` |
| `buffer_size` | Rolling frame buffer size for live demo | `64` |
| `fps_display` | Show FPS counter on live demo overlay | `true` |

**Logging:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_wandb` | Enable Weights & Biases logging | `false` |
| `use_tensorboard` | Enable TensorBoard logging | `true` |
| `wandb_project` | W&B project name | `wlasl-recognition` |
| `wandb_run_name` | W&B run name (auto-generated if not set) | `null` |
| `log_interval` | Steps between log entries | `10` |

**Resume:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resume_checkpoint` | Path to checkpoint for resuming training | `null` |

`num_classes` is **always** auto-derived from `wlasl_variant` (100 → 100, 300 → 300, etc.). Any explicit `num_classes` in the YAML is silently overridden — do not set it manually.

---

## Approach Details

### Approach A: Pose/Keypoint Transformer

The recommended starting approach. MediaPipe Holistic extracts 543 landmarks per frame (33 pose + 21 left hand + 21 right hand + 468 face), centered on the shoulder midpoint and scaled by shoulder width. When `use_motion: true` (default), frame-to-frame velocity is concatenated with position, producing 6 features per keypoint `(x, y, z, dx, dy, dz)`.

**Advantages:** Lightweight, fast inference, background-invariant.

**Pipeline:** Video → MediaPipe → Shoulder-Centered Normalization → Motion Features → Augment → Transformer → Softmax

**Data augmentation pipeline** (training only):
- Temporal speed perturbation (0.8x–1.2x)
- Random temporal crop to T frames
- Keypoint rotation (up to 15 degrees)
- Keypoint translation (up to 0.1 shift)
- Keypoint horizontal flip with landmark swapping
- Keypoint dropout (frame-level and landmark-level)
- Keypoint noise (sigma=0.02)
- Random scaling (0.9x–1.1x)

**Regularization:** Mixup interpolation (`mixup_alpha: 0.2`), label smoothing, dropout, and weighted sampling for class imbalance.

### Approach B: RGB Video Classifier

Uses pretrained 3D CNN backbones to classify raw video frames.

**Supported backbones:** `r2plus1d_18`, `r3d_18`, `mc3_18` (torchvision), `slow_r50`, `slowfast_r50`, `x3d_m` (pytorchvideo)

**Advantages:** Captures full visual context including texture and fine finger details.

### Approach C: Hybrid Fusion

Combines Approaches A and B. Two fusion modes:
- **`concat`** — Concatenate pose and video feature vectors before the classification head.
- **`attention`** — Cross-attention between pose tokens and video spatial features.

---

## Troubleshooting

### HTML files masquerading as videos
Expired WLASL URLs often return an HTML lander page (saved as `.mp4`) rather than a 404. Run `scripts/validate_videos.py --delete` before preprocessing to clean these up (see [Step 2](#2-download-the-dataset)).

### MediaPipe installation issues
- Compatible with Python 3.9–3.12.
- On macOS with Apple Silicon: `pip install mediapipe-silicon`
- **Run the diagnostic script** to verify your installation:
  ```bash
  python scripts/check_mediapipe.py
  ```
  This prints your mediapipe version, available modules, and whether the `Holistic` model is accessible.
- If you see `AttributeError: module 'mediapipe' has no attribute 'solutions'`, your MediaPipe version is incompatible. This commonly happens on **Windows with Python 3.12** and newer mediapipe builds. Fix with: `pip install --force-reinstall mediapipe==0.10.11` (or `mediapipe-silicon` on Apple Silicon). The code has a fallback import path (`mediapipe.python.solutions`) that resolves this for some versions automatically.
- Zero-padded keypoints for frames where detection fails are handled automatically.
- Preprocessing uses `spawn` multiprocessing context (not `fork`) to avoid MediaPipe crashes on macOS.

### CUDA out of memory
- Reduce `batch_size` (try 8 or 4).
- Enable `fp16: true`.
- Reduce `T` (e.g., 32 instead of 64).
- For video models, set `image_size: 112`.

### Webcam not detected
- Try `--camera 1` or `--camera 2`.
- Linux: check `ls -la /dev/video*`.
- macOS: grant camera access in System Settings → Privacy & Security → Camera.
- Windows: check Device Manager → Cameras. Grant camera access in Settings → Privacy & security → Camera.

### Low accuracy
- Check split CSV row counts to ensure enough training videos were downloaded.
- Enable `weighted_sampling: true` for class-imbalanced subsets.
- Run the error analysis notebook (`notebooks/03_error_analysis.ipynb`) to find confused class pairs.

### Diagnosing partial data (most common issue)

Many WLASL URLs have expired, so you will likely end up with far fewer usable videos than the annotation file lists. This is the single biggest factor in accuracy. Check your effective dataset size:

**Linux/macOS:**

```bash
# Row counts in split CSVs (includes videos you may not have)
wc -l data/splits/WLASL100/*.csv

# How many .npy keypoint files were actually produced
ls data/processed/*.npy | wc -l
```

**Windows (PowerShell):**

```powershell
# Row counts in split CSVs
Get-ChildItem data\splits\WLASL100\*.csv | ForEach-Object { Write-Host "$($_.Name): $((Get-Content $_).Count) lines" }

# How many .npy keypoint files were actually produced
(Get-ChildItem data\processed\*.npy).Count
```

**Cross-platform (Python):**

```bash
# Effective training samples (rows in CSV that have matching .npy files)
python -c "
import pandas as pd; from pathlib import Path
train = pd.read_csv('data/splits/WLASL100/train.csv')
npy = Path('data/processed')
eff = train[train['video_id'].apply(lambda v: (npy/f'{v}.npy').exists())]
counts = eff['label_idx'].value_counts()
print(f'Effective train: {len(eff)} samples, {eff[\"label_idx\"].nunique()} classes')
print(f'Samples/class: min={counts.min()}, mean={counts.mean():.1f}, max={counts.max()}')
print(f'Classes with <=2 samples: {(counts<=2).sum()}')
"
```

**If effective samples < 500:** Training will be very challenging. The default `configs/pose_transformer.yaml` is already tuned for this scenario (high dropout, weighted sampling, low LR). Expect 30–50% top-1 accuracy.

**To get more data:**
1. Use the Kaggle download script for the full ~12K video archive: `python scripts/download_kaggle.py`
2. Re-run preprocessing after adding new videos — already-processed `.npy` files are skipped automatically.
3. Try `--subset WLASL300` to include more glosses (you may have videos for classes outside WLASL100).

### `wlasl_variant` / `num_classes` mismatch

`num_classes` is always auto-derived from `wlasl_variant`. If your YAML says `wlasl_variant: 300` but you only preprocessed WLASL100, training will fail because `data/splits/WLASL300/train.csv` does not exist. Make sure `wlasl_variant` in your config matches the subset you preprocessed.

---

## Recommended Configurations

These are tuned starting points for each dataset variant. Copy the base config and modify:

### WLASL100 (recommended starting point)

~2,000 annotations, ~100 glosses. Expect 400–1,200 usable training samples depending on download availability.

```yaml
approach: pose_transformer
wlasl_variant: 100
T: 64
use_motion: true            # velocity features (position + frame differences)
d_model: 256
nhead: 8
num_layers: 4
dropout: 0.3
batch_size: 32
lr: 1.0e-3
scheduler: onecycle
warmup_epochs: 10
label_smoothing: 0.1
mixup_alpha: 0.2            # mixup regularization
weighted_sampling: true     # important — classes are imbalanced
early_stopping_patience: 20
epochs: 100
```

With very few training videos (<500 usable), increase regularization:

```yaml
dropout: 0.4
label_smoothing: 0.15
mixup_alpha: 0.3
batch_size: 16
```

### WLASL300

~5,000 annotations, 300 glosses. More data per class on average.

```yaml
approach: pose_transformer
wlasl_variant: 300
T: 64
d_model: 256
nhead: 8
num_layers: 6           # deeper than WLASL100
dropout: 0.25
batch_size: 32
lr: 5.0e-4
scheduler: cosine
warmup_epochs: 10
label_smoothing: 0.1
weighted_sampling: true
early_stopping_patience: 25
epochs: 150
```

### WLASL1000 / WLASL2000

Much larger class count. Needs more model capacity and training time.

```yaml
approach: pose_transformer
wlasl_variant: 1000       # or 2000
T: 64
d_model: 384              # wider
nhead: 8
num_layers: 6
dropout: 0.2
batch_size: 64
lr: 5.0e-4
scheduler: cosine
warmup_epochs: 15
label_smoothing: 0.1
weighted_sampling: true
early_stopping_patience: 30
epochs: 200
```

For WLASL2000, consider the video approach (Approach B) or fusion (Approach C) — the added visual detail helps disambiguate the larger vocabulary.

### Video Classifier (Approach B)

Use when you have sufficient GPU memory and want to leverage pretrained RGB features.

```yaml
approach: video
backbone: r2plus1d_18
pretrained: true
T: 32                     # video models are memory-heavy, keep T lower
image_size: 224            # reduce to 112 if GPU memory is tight
batch_size: 8              # 3D CNNs need small batches
lr: 1.0e-4                 # lower LR for finetuning pretrained backbone
dropout: 0.4
fp16: true                 # essential for video models
```

### Fusion (Approach C)

Combines Approaches A and B for highest accuracy. Requires both keypoints and raw videos.

```yaml
approach: fusion
fusion: concat             # start with concat, try attention if concat plateaus
fusion_dim: 256
backbone: r2plus1d_18
pretrained: true
T: 64
batch_size: 8
lr: 1.0e-4
dropout: 0.3
fp16: true
```

---

## Tips & Best Practices

### Hardware-Specific Setup

**CPU-only (no GPU)**
```yaml
fp16: false                # FP16 only works on CUDA
batch_size: 8              # smaller batches to avoid memory pressure
num_workers: 2
```
Stick to Approach A (pose_transformer). Video models are too slow on CPU for training (inference is manageable).

**GPU / CUDA**
```yaml
fp16: true
batch_size: 32             # increase to fill GPU memory (or 64 for large GPUs)
num_workers: 4
```
Monitor GPU memory with `nvidia-smi`. If you run out of memory, reduce `batch_size` first, then `T`, then `image_size` (for video models).

**Apple Silicon (M1/M2/M3)**
- Install MediaPipe: `pip install mediapipe-silicon`
- MPS backend is supported by PyTorch but gains over CPU are inconsistent for these model sizes. Use `--device cpu` for the live demo to avoid MPS overhead.
- Preprocessing already uses `spawn` multiprocessing context to avoid macOS fork crashes.

### Training with Limited Data

WLASL's expired URLs mean you may only get 30–60% of the annotated videos. When your training set is small (<500 samples for 100 classes):

1. **Enable weighted sampling** (`weighted_sampling: true`) — ensures every class is seen equally despite imbalance.
2. **Increase dropout** to 0.4–0.5 to reduce overfitting.
3. **Increase label smoothing** to 0.15–0.2 for better calibration.
4. **Use smaller batch sizes** (8–16) so the model sees more update steps per epoch.
5. **Lower the learning rate** to 5e-4 or 3e-4 with cosine scheduler.
6. **Try BiLSTM** (`approach: pose_bilstm`) — fewer parameters, less prone to overfitting on tiny datasets.
7. **Download from Kaggle** (`python scripts/download_kaggle.py`) — the full ~12K video archive is available as a single download.

### Improving Accuracy

- **Start with Approach A** (pose_transformer). It trains fastest and is easiest to debug.
- **Enable motion features** (`use_motion: true`) — velocity information captures signing dynamics and typically adds 5–8% accuracy.
- **Use mixup** (`mixup_alpha: 0.2`) — regularizes training by interpolating between random sample pairs.
- **Enable TTA for evaluation** (`use_tta: true`) — averages predictions over original + horizontally flipped input for 2–4% evaluation boost.
- **Use the error analysis notebook** (`notebooks/03_error_analysis.ipynb`) to find which classes are confused, then inspect those videos manually.
- **Try the cosine scheduler** (`scheduler: cosine`) if onecycle doesn't converge well — cosine with warm-up is often more stable.
- **Increase sequence length** (`T: 96` or `T: 128`) if signs in your dataset are long — some signs take 3+ seconds at 25 fps.
- **Scale up to fusion** (Approach C) once you've maxed out Approach A's accuracy — it typically adds 3–8% top-1 over pose-only.

### What to Expect

Realistic accuracy ranges depend heavily on how many videos you have:

| Dataset | Training Samples | Approach A (expected) | Approach C (expected) |
|---------|------------------|-----------------------|-----------------------|
| WLASL100 | 400–800 | 45–60% top-1 | 55–68% top-1 |
| WLASL100 | 800–1,500 | 60–70% top-1 | 68–78% top-1 |
| WLASL300 | 2,000–4,000 | 45–55% top-1 | 55–65% top-1 |

If your val loss is around `ln(num_classes)` (e.g., 4.6 for 100 classes), the model is near random — check that enough training samples are being loaded.

### Common Pitfalls

1. **HTML videos**: Run `scripts/validate_videos.py --delete` before preprocessing. Expired WLASL URLs return HTML pages saved as `.mp4` files.
2. **Corrupt videos**: Some downloads are truncated (OpenCV reports "moov atom not found"). The preprocessing pipeline skips these automatically, but they inflate your file count.
3. **Wrong video count**: The official WLASL download scripts fetch ALL ~21,000 videos (all 2,000 glosses), not just your target variant. The preprocessing pipeline filters to the correct subset.
4. **MediaPipe warnings**: `inference_feedback_manager.cc` warnings are harmless TFLite logs. Suppress with `GLOG_minloglevel=2 python ...` (Linux/macOS) or `$env:GLOG_minloglevel=2; python ...` (Windows PowerShell) or `set GLOG_minloglevel=2 && python ...` (Windows cmd).
5. **Empty val set**: If the val split CSV has very few rows, some classes may have zero val samples. This makes early stopping and accuracy metrics unreliable — check row counts after preprocessing: `wc -l data/splits/WLASL100/*.csv` (Linux/macOS) or `Get-ChildItem data\splits\WLASL100\*.csv | ForEach-Object { Write-Host "$($_.Name): $((Get-Content $_).Count)" }` (Windows PowerShell).
6. **OneCycleLR NaN loss**: If you resume training from a checkpoint with a different total step count, the scheduler can go out of range. Start fresh or use `scheduler: cosine` for resumed runs.

---

## Recommended Library & CUDA Versions

This project requires `torch>=2.1.0,<2.5.0`. The table below shows which CUDA toolkit versions are compatible with each PyTorch release, and the matching torchvision / torchaudio versions.

### PyTorch ↔ CUDA Compatibility

| PyTorch | torchvision | torchaudio | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 | Install command |
|---------|-------------|------------|-----------|-----------|-----------|-----------------|
| 2.4.1 | 0.19.1 | 2.4.1 | Yes | Yes | Yes | `pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124` |
| 2.3.1 | 0.18.1 | 2.3.1 | Yes | Yes | No | `pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121` |
| 2.2.2 | 0.17.2 | 2.2.2 | Yes | Yes | No | `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121` |
| 2.1.2 | 0.16.2 | 2.1.2 | Yes | Yes | No | `pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121` |

**How to choose:**
- **CUDA 12.4** — Use PyTorch 2.4.x. Best for RTX 30xx/40xx/50xx with recent drivers.
- **CUDA 12.1** — Supported by all versions above. Safe default for most setups.
- **CUDA 11.8** — Supported by all versions above. Use if your driver is older or you're on a shared cluster.
- **CPU only** — Append `--index-url https://download.pytorch.org/whl/cpu` to any install command.

Check your CUDA version with `nvidia-smi` (top-right corner shows the maximum CUDA version your driver supports).

> **Note:** The `--index-url` flag must match your CUDA version, not your PyTorch version. If you install the wrong CUDA variant, `torch.cuda.is_available()` will return `False`.

### MediaPipe

| Platform | Package | Version Range | Install command |
|----------|---------|---------------|-----------------|
| Linux / Windows | `mediapipe` | `>=0.10.7,<=0.10.14` | `pip install mediapipe` (included in requirements.txt) |
| macOS (Apple Silicon) | `mediapipe-silicon` | `>=0.10.7` | `pip install mediapipe-silicon` |
| macOS (Intel) | `mediapipe` | `>=0.10.7,<=0.10.14` | `pip install mediapipe` |

- MediaPipe is compatible with **Python 3.9–3.12**.
- On Apple Silicon, the standard `mediapipe` package may fail to install. Use `mediapipe-silicon` instead — it provides the same API.
- On **Windows with Python 3.12**, some mediapipe versions expose `solutions` under `mediapipe.python.solutions` instead of `mediapipe.solutions`. If you hit this issue, pin to `mediapipe==0.10.11`.
- Both packages provide `mediapipe.solutions.holistic` used by the preprocessing pipeline.

### Other Key Libraries

| Library | Required Version | Notes |
|---------|-----------------|-------|
| `opencv-python` | `>=4.8.0,<4.11.0` | Video I/O and frame capture. Webcam access requires system camera permissions. |
| `numpy` | `>=1.24.0,<2.1.0` | NumPy 2.x introduced breaking changes — stay below 2.1 for compatibility with all dependencies. |
| `onnxruntime` | `>=1.16.0,<1.20.0` | For ONNX export verification. Use `onnxruntime-gpu` instead if you want GPU-accelerated ONNX inference. |
| `pytorchvideo` | `>=0.1.5,<0.2.0` | SlowFast and X3D backbones (Approach B). Only needed if using video models. |
| `albumentations` | `>=1.3.1,<1.5.0` | Image augmentations for video frame preprocessing (Approach B/C). |
| `kaggle` | `>=1.6.0,<1.8.0` | Kaggle API for dataset download. Only needed if using `scripts/download_kaggle.py`. |

For the full list of dependencies with version ranges, see [`requirements.txt`](requirements.txt).

Source: [PyTorch CUDA Compatibility Matrix](https://github.com/eminsafa/pytorch-cuda-compatibility) | [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)

---

## Citation

```bibtex
@inproceedings{li2020word,
  title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={1459--1469},
  year={2020}
}
```

## License

This project is for educational and research purposes. The WLASL dataset has its own licensing terms — check the [official repository](https://github.com/dxli94/WLASL) before commercial use.
