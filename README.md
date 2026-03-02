# Live American Sign Language Recognition using WLASL

A real-time ASL word-level recognition system that captures live webcam video, processes hand/body movements frame-by-frame using MediaPipe, and predicts the signed word using deep learning models trained on the [WLASL dataset](https://github.com/dxli94/WLASL).

## Architecture

```
                        APPROACH A (Pose/Keypoint)
                    +-------------------------------+
                    |  MediaPipe     Transformer/   |
Webcam  --> Frame --+  Holistic  --> LSTM Encoder --+--> Predicted
Feed        Buffer  |  Keypoints    (T, 543*3)     |    Gloss +
(OpenCV)    (T=64)  +-------------------------------+    Confidence
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
| A - Pose Transformer | Transformer Encoder / BiLSTM | MediaPipe keypoints (T, 543, 3) | 60–70% |
| B - Video Classifier | R(2+1)D-18, R3D-18, SlowFast | RGB frames (3, T, 224, 224) | 65–75% |
| C - Hybrid Fusion | Concat / Cross-Attention fusion of A+B | Both streams | 70–78% |

---

## Quick Start

### 1. Environment Setup

```bash
git clone <this-repo-url>
cd "Live American Sign Language Recognition using WLASL"

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

---

### 2. Download the Dataset

```bash
# Download the WLASL annotation JSON and print video download instructions
python scripts/download_wlasl.py --subset WLASL100
```

This creates `data/annotations/WLASL_v0.3.json` and the directory structure under `data/`. Follow the printed instructions to download the actual video files from the [official WLASL repo](https://github.com/dxli94/WLASL) or a community mirror, then place them in `data/raw/`.

#### Validate downloaded videos

Many WLASL URLs have expired. When a URL is dead, servers often return an HTML redirect page (saved as `.mp4`) instead of a 404. Run the validator before preprocessing to remove these fake files:

```bash
# Report how many invalid files exist
python scripts/validate_videos.py --video-dir data/raw

# Delete invalid files (HTML redirects and empty files)
python scripts/validate_videos.py --video-dir data/raw --delete

# Delete and save a list of valid video IDs for reference
python scripts/validate_videos.py --video-dir data/raw --delete --save-valid data/valid_ids.txt
```

The validator checks the first 256 bytes of each file for HTML signatures (`<!DOCTYPE html>`, `<html>`, etc.) and reports counts of valid / HTML / empty files. The preprocessing pipeline also skips unreadable files automatically, but cleaning them up first saves processing time.

---

### 3. Preprocess Data

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
# Copy and modify
cp configs/pose_transformer.yaml configs/pose_wlasl300.yaml
# Edit wlasl_variant: 300 and num_classes: 300 in the new file
python -m src.training.train --config configs/pose_wlasl300.yaml
```

---

### 5. Evaluate

```bash
python -m src.training.evaluate \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt \
    --split val \
    --output-dir eval_results
```

This prints top-1/top-5 accuracy, per-class breakdown, and saves a confusion matrix heatmap to `eval_results/`.

---

### 6. Run the Live Demo

```bash
python -m src.inference.live_demo \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt \
    --camera 0 \
    --device cpu
```

**Controls:**
- `q` — quit
- `s` — save the current prediction to a log file

The demo runs three threads: a capture thread reads webcam frames continuously, an inference thread runs the model every 0.5 s on a rolling buffer of T frames, and the main thread renders the overlay. Predictions are smoothed over the last 5 inference windows and only displayed when confidence exceeds the configured threshold (default: 0.6).

---

### 7. Single Video Prediction

```bash
python -m src.inference.predict \
    --video path/to/video.mp4 \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt
```

Returns the predicted gloss, confidence score, and top-5 alternatives.

---

### 8. Export to ONNX

```bash
python -m src.inference.export_onnx \
    --config configs/pose_transformer.yaml \
    --checkpoint checkpoints/best_model.pt \
    --output model.onnx \
    --verify \
    --benchmark
```

`--verify` runs a forward pass through ONNX Runtime to confirm output shapes match. `--benchmark` measures average inference latency over 100 runs.

---

## Project Structure

```
.
├── configs/
│   ├── pose_transformer.yaml    # Approach A defaults
│   ├── video_classifier.yaml    # Approach B defaults
│   └── fusion.yaml              # Approach C defaults
├── data/
│   ├── raw/                     # Downloaded video files (all variants share this)
│   ├── processed/               # Extracted keypoints as .npy (shared across variants)
│   ├── annotations/             # WLASL JSON annotation file
│   └── splits/
│       ├── WLASL100/            # train/val/test CSVs for 100-class variant
│       ├── WLASL300/            # train/val/test CSVs for 300-class variant
│       └── ...                  # one subdirectory per variant
├── src/
│   ├── data/
│   │   ├── preprocess.py        # Download, parse, extract keypoints
│   │   ├── augment.py           # Temporal & spatial augmentations
│   │   └── dataset.py           # PyTorch Dataset classes
│   ├── models/
│   │   ├── pose_transformer.py  # Transformer & BiLSTM (Approach A)
│   │   ├── video_i3d.py         # 3D CNN backbones (Approach B)
│   │   └── fusion.py            # Multi-modal fusion (Approach C)
│   ├── training/
│   │   ├── config.py            # Config dataclass + YAML serialization
│   │   ├── train.py             # Training loop
│   │   └── evaluate.py          # Metrics & confusion matrix
│   └── inference/
│       ├── predict.py           # Single-video prediction
│       ├── export_onnx.py       # ONNX export & latency benchmark
│       └── live_demo.py         # Real-time webcam demo
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_keypoint_visualization.ipynb
│   └── 03_error_analysis.ipynb
├── scripts/
│   ├── download_wlasl.py        # Download annotations, print video instructions
│   └── validate_videos.py       # Detect and remove HTML-disguised video files
├── checkpoints/                 # Saved model weights
├── logs/                        # TensorBoard training logs
└── requirements.txt
```

---

## Configuration Guide

All hyperparameters live in YAML files under `configs/`. Key settings:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `approach` | `pose_transformer`, `pose_bilstm`, `video`, `fusion` | `pose_transformer` |
| `wlasl_variant` | Dataset size: `100`, `300`, `1000`, `2000` | `100` |
| `num_classes` | Auto-derived from `wlasl_variant` — do not set manually | `100` |
| `T` | Temporal sequence length in frames | `64` |
| `d_model` | Transformer embedding dimension | `256` |
| `nhead` | Number of attention heads | `8` |
| `num_layers` | Number of encoder layers | `4` |
| `batch_size` | Training batch size | `16` |
| `lr` | Learning rate | `5e-4` |
| `epochs` | Maximum training epochs | `100` |
| `fp16` | Mixed-precision (FP16) training | `true` |
| `weighted_sampling` | Weighted sampler to counter class imbalance | `true` |
| `scheduler` | LR scheduler: `onecycle` or `cosine` (warmup + cosine annealing) | `cosine` |
| `confidence_threshold` | Minimum confidence for live display | `0.6` |
| `smoothing_window` | Number of inference windows to smooth predictions over | `5` |

`num_classes` is **always** auto-derived from `wlasl_variant` (100 → 100, 300 → 300, etc.). Any explicit `num_classes` in the YAML is silently overridden — do not set it manually.

---

## Approach Details

### Approach A: Pose/Keypoint Transformer

The recommended starting approach. MediaPipe Holistic extracts 543 landmarks per frame (33 pose + 21 left hand + 21 right hand + 468 face), normalized relative to the nose and shoulder width, then fed into a Transformer encoder or BiLSTM.

**Advantages:** Lightweight, fast inference, background-invariant.

**Pipeline:** Video → MediaPipe → Normalize → Temporal Crop/Augment → Transformer → Softmax

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

### Low accuracy
- Check split CSV row counts to ensure enough training videos were downloaded.
- Enable `weighted_sampling: true` for class-imbalanced subsets.
- Run the error analysis notebook (`notebooks/03_error_analysis.ipynb`) to find confused class pairs.

### Diagnosing partial data (most common issue)

Many WLASL URLs have expired, so you will likely end up with far fewer usable videos than the annotation file lists. This is the single biggest factor in accuracy. Check your effective dataset size:

```bash
# Row counts in split CSVs (includes videos you may not have)
wc -l data/splits/WLASL100/*.csv

# How many .npy keypoint files were actually produced
ls data/processed/*.npy | wc -l

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
1. Search Kaggle for "WLASL" dataset mirrors.
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
d_model: 256
nhead: 8
num_layers: 4
dropout: 0.3
batch_size: 32
lr: 1.0e-3
scheduler: onecycle
warmup_epochs: 10
label_smoothing: 0.1
weighted_sampling: true   # important — classes are imbalanced
early_stopping_patience: 20
epochs: 100
```

With very few training videos (<500 usable), increase regularization:

```yaml
dropout: 0.4
label_smoothing: 0.15
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
7. **Check Kaggle mirrors** for community-uploaded WLASL video archives.

### Improving Accuracy

- **Start with Approach A** (pose_transformer). It trains fastest and is easiest to debug.
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
4. **MediaPipe warnings**: `inference_feedback_manager.cc` warnings are harmless TFLite logs. Suppress with `GLOG_minloglevel=2 python ...` if they clutter your output.
5. **Empty val set**: If the val split CSV has very few rows, some classes may have zero val samples. This makes early stopping and accuracy metrics unreliable — check `wc -l data/splits/WLASL100/*.csv` after preprocessing.
6. **OneCycleLR NaN loss**: If you resume training from a checkpoint with a different total step count, the scheduler can go out of range. Start fresh or use `scheduler: cosine` for resumed runs.

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
