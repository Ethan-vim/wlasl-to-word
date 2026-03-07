# Project Structure & Workflow

This document maps the entire pipeline from data download to live inference, showing which files call which and how data flows through the system.

---

## End-to-End Workflow

```
 PHASE 1: DATA                PHASE 2: TRAINING              PHASE 3: USAGE
 ─────────────                ──────────────────              ──────────────

 download_wlasl.py ─┐
                    ├─> data/raw/*.mp4
 download_kaggle.py ┘         │
                              v
              validate_videos.py
                              │
                              v
                      preprocess.py
                     ┌────────┴────────┐
                     v                 v
            data/processed/    data/splits/WLASL{N}/
            *.npy (keypoints)  train.csv, val.csv, test.csv
                     │                 │
                     └────────┬────────┘
                              v
                    ┌─── train.py ───┐         evaluate.py
                    │   (loads data, │              │
                    │    builds      │              v
                    │    model,      │        eval_results/
                    │    trains)     │        confusion_matrix.png
                    │               │
                    v               v
              checkpoints/    logs/
              best_model.pt   tensorboard/
                    │
         ┌──────────┼──────────┐
         v          v          v
    predict.py  live_demo.py  export_onnx.py
    (single      (webcam       (ONNX
     video)       real-time)    export)
```

---

## File Dependency Graph

Shows which project files each module imports from (`src.*` imports only).

```
                         ┌──────────────────┐
                         │  config.py       │
                         │  (Config,        │
                         │   load_config,   │
                         │   save_config)   │
                         └───────┬──────────┘
                   ┌─────────────┼─────────────────────────────┐
                   │             │             │               │
                   v             v             v               v
            ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐
            │ train.py │  │evaluate. │  │ predict.  │  │ live_demo. │
            │          │  │  py      │  │   py      │  │    py      │
            └──┬───┬───┘  └──┬───┬──┘  └──┬──┬──┬──┘  └──┬──┬─────┘
               │   │         │   │        │  │  │        │  │
          ┌────┘   │    ┌────┘   │        │  │  │        │  │
          v        v    v        v        │  │  │        │  │
    ┌──────────┐ ┌──────────┐             │  │  │        │  │
    │augment.py│ │dataset.py│             │  │  │        │  │
    └──────────┘ └──────────┘             │  │  │        │  │
                                          │  │  │        │  │
               ┌──────────────────────────┘  │  │        │  │
               v                             v  │        v  │
    ┌────────────────┐            ┌─────────────────┐      │
    │ preprocess.py  │            │pose_transformer.│      │
    │ (normalize,    │            │   py            │      │
    │  keypoints)    │            │ (PoseTransformer│      │
    └────────────────┘            │  PoseBiLSTM)    │      │
                                  └─────────────────┘      │
                                                           │
               ┌───────────────────────────────────────────┘
               v
    ┌─────────────────┐    ┌─────────────────┐
    │  video_i3d.py   │    │   fusion.py     │
    │ (VideoClassifier│    │ (FusionModel,   │
    │  build_video_   │    │  CrossAttention │
    │  model)         │    │  Fusion)        │
    └─────────────────┘    └─────────────────┘
```

---

## Module Details

### Scripts (Entry Points)

| Script | Purpose | Imports From | Output |
|--------|---------|--------------|--------|
| `scripts/download_wlasl.py` | Download annotations + print video instructions | `src.data.preprocess` | `data/annotations/`, `data/raw/` |
| `scripts/download_kaggle.py` | Download full video archive from Kaggle | `src.data.preprocess` | `data/raw/*.mp4` |
| `scripts/validate_videos.py` | Detect and remove fake HTML video files | (none) | Cleaned `data/raw/` |
| `scripts/reset_configs.py` | Reset YAML configs to README defaults | (none) | `configs/*.yaml` |
| `scripts/check_mediapipe.py` | Verify MediaPipe installation, diagnose `solutions` import issues | (none) | Diagnostic output to stdout |
| `scripts/auto_config.py` | Auto-detect hardware (CUDA/MPS/CPU) and generate optimized configs | (none) | `configs/*.yaml` |

### Data Pipeline (`src/data/`)

| Module | Key Functions / Classes | Used By |
|--------|------------------------|---------|
| `preprocess.py` | `download_wlasl_annotations()`, `parse_wlasl_annotations()`, `extract_keypoints_mediapipe()`, `normalize_keypoints()`, `preprocess_dataset()`, `create_splits()` | `download_wlasl.py`, `download_kaggle.py`, `predict.py`, `live_demo.py` |
| `augment.py` | `TemporalCrop`, `TemporalSpeedPerturb`, `KeypointHorizontalFlip`, `KeypointRotation`, `KeypointTranslation`, `KeypointDropout`, `KeypointNoise`, `KeypointScale`, `Compose`, `get_train_transforms()`, `get_val_transforms()` | `train.py`, `evaluate.py`, `predict.py` |
| `dataset.py` | `WLASLKeypointDataset`, `WLASLVideoDataset`, `WLASLFusionDataset`, `get_dataloader()` | `train.py`, `evaluate.py` |

### Models (`src/models/`)

| Module | Key Classes | Build Function | Approaches |
|--------|-------------|----------------|------------|
| `pose_transformer.py` | `PoseTransformer`, `PoseBiLSTM` | `build_pose_model(cfg)` | A (pose_transformer, pose_bilstm) |
| `video_i3d.py` | `VideoClassifier` | `build_video_model(cfg)` | B (video) |
| `fusion.py` | `FusionModel`, `CrossAttentionFusion` | `build_fusion_model(cfg)` | C (fusion) |

All `build_*_model()` functions take a `Config` object and return an `nn.Module`.

### Training (`src/training/`)

| Module | Key Functions | Imports From |
|--------|---------------|--------------|
| `config.py` | `Config` (dataclass), `load_config()`, `save_config()` | (none — leaf dependency) |
| `train.py` | `train_one_epoch()`, `validate()`, `main()` | `config`, `augment`, `dataset`, `pose_transformer`, `video_i3d`, `fusion` |
| `evaluate.py` | `compute_metrics()`, `plot_confusion_matrix()`, `find_hard_negatives()`, `evaluate_latency()`, `main()` | `config`, `augment`, `dataset`, `pose_transformer`, `video_i3d`, `fusion` |

### Inference (`src/inference/`)

| Module | Key Classes / Functions | Imports From |
|--------|------------------------|--------------|
| `predict.py` | `SignPredictor`, `_load_class_names()` | `config`, `augment`, `preprocess`, `pose_transformer`, `video_i3d` |
| `live_demo.py` | `FrameBuffer`, `LivePredictor`, `ASLDisplay`, `run_demo()` | `config`, `preprocess`, `pose_transformer` |
| `export_onnx.py` | `export_to_onnx()`, `verify_onnx()`, `benchmark_onnx()` | `config`, `pose_transformer`, `video_i3d` |

---

## Data Flow Diagrams

### Training Data Flow

```
data/raw/*.mp4
       │
       v
  ┌─────────────────────────────────┐
  │  preprocess.py                  │
  │  extract_keypoints_mediapipe()  │
  │  normalize_keypoints()          │  ──> data/processed/*.npy  (T, 543, 3)
  │  create_splits()                │  ──> data/splits/WLASL{N}/train.csv
  └─────────────────────────────────┘

data/processed/*.npy + data/splits/WLASL{N}/train.csv
       │
       v
  ┌─────────────────────────────────┐
  │  dataset.py                     │
  │  WLASLKeypointDataset           │
  │    __getitem__():               │
  │      load .npy                  │
  │      pad/crop to T frames      │
  │      compute velocity (motion)  │  ──> (T, 543*6) when use_motion=True
  │      apply augmentations        │
  │      flatten to (T, input_dim)  │
  └─────────────────────────────────┘
       │
       v
  ┌─────────────────────────────────┐
  │  train.py                       │
  │  train_one_epoch():             │
  │    mixup (if enabled)           │
  │    forward pass through model   │
  │    loss + backprop              │
  │  validate():                    │
  │    forward pass (no augment)    │
  │    compute top-1 / top-5 acc    │  ──> checkpoints/best_model.pt
  │    early stopping check         │  ──> logs/ (TensorBoard)
  └─────────────────────────────────┘
```

### Inference Data Flow

```
Single Video (predict.py):

  video.mp4 ──> MediaPipe ──> normalize ──> velocity ──> model ──> top-5 predictions
       │                                                   ^
       │         OR                                        │
  keypoints.npy ──────────────> velocity ──────────────────┘


Live Demo (live_demo.py):

  Webcam ──> FrameBuffer(T=64) ──> MediaPipe ──> normalize ──> velocity ──> model
    │              │                                                          │
    │         rolls every                                                     v
    │         0.5 seconds                                                predictions
    │                                                                         │
    v                                                                         v
  Display <───── overlay predicted gloss + confidence (smoothed over 5 windows)


ONNX Export (export_onnx.py):

  checkpoint ──> load model ──> torch.onnx.export() ──> model.onnx
                                     │
                              verify (optional) ──> ONNX Runtime forward pass
                              benchmark (optional) ──> avg latency over 100 runs
```

### Model Architecture Flow (Approach A)

```
Input: (batch, T, input_dim)     input_dim = 543*6 = 3258 (with motion)
              │
              v
    ┌─────────────────────┐
    │  Linear projection  │  (input_dim -> d_model)
    └─────────┬───────────┘
              v
    ┌─────────────────────┐
    │  Positional Encoding│  (learned, T positions)
    └─────────┬───────────┘
              v
    ┌─────────────────────┐
    │  TransformerEncoder │  (num_layers=4, nhead=8, d_model=256)
    │  x4 encoder layers  │
    └─────────┬───────────┘
              v
    ┌─────────────────────┐
    │  Mean pooling        │  (T, d_model) -> (d_model,)
    └─────────┬───────────┘
              v
    ┌─────────────────────┐
    │  Classification head│  Linear(d_model -> num_classes)
    └─────────┬───────────┘
              v
    Output: (batch, num_classes) logits
```

---

## Configuration Flow

```
configs/pose_transformer.yaml
configs/video_classifier.yaml        ──> load_config() ──> Config dataclass
configs/fusion.yaml                                             │
                                                    ┌───────────┼───────────┐
                                                    v           v           v
                                              train.py    evaluate.py  predict.py
                                                                        live_demo.py
                                                                        export_onnx.py

Config.__post_init__() auto-derives:
    wlasl_variant: 100  ──>  num_classes: 100
    wlasl_variant: 300  ──>  num_classes: 300
    wlasl_variant: 1000 ──>  num_classes: 1000
    wlasl_variant: 2000 ──>  num_classes: 2000
```

---

## Tests (`tests/`)

Each test file maps to one or more source modules:

| Test File | Tests For |
|-----------|-----------|
| `test_augment.py` | `src/data/augment.py` — all transform classes and pipeline presets |
| `test_config.py` | `src/training/config.py` — defaults, load/save, YAML roundtrip |
| `test_dataset.py` | `src/data/dataset.py` — Dataset, DataLoader, pad/crop, motion features |
| `test_evaluate.py` | `src/training/evaluate.py` — metrics, TTA, hard negatives, latency |
| `test_export_onnx.py` | `src/inference/export_onnx.py` — ONNX export and verification |
| `test_live_demo.py` | `src/inference/live_demo.py` — FrameBuffer, prediction smoothing |
| `test_models.py` | `src/models/` — PoseTransformer, PoseBiLSTM, FusionModel shapes |
| `test_predict.py` | `src/inference/predict.py` — SignPredictor inference paths |
| `test_preprocess.py` | `src/data/preprocess.py` — normalization, annotation parsing, splits |
| `test_train.py` | `src/training/train.py` — accuracy, mixup helpers |
| `test_dependencies.py` | All `requirements.txt` libraries — version checks, feature compatibility, src module imports (110 tests) |

All tests use `conftest.py` shared fixtures (tmp datasets, keypoint generators) and are fully isolated (no project data or checkpoints needed).

---

## CLI Entry Points

| Command | Module | What It Does |
|---------|--------|-------------|
| `python scripts/download_wlasl.py` | `scripts/download_wlasl.py` | Download annotations, print video download instructions |
| `python scripts/download_kaggle.py` | `scripts/download_kaggle.py` | Download all videos from Kaggle |
| `python scripts/validate_videos.py` | `scripts/validate_videos.py` | Scan and clean fake video files |
| `python scripts/reset_configs.py` | `scripts/reset_configs.py` | Reset configs to recommended defaults |
| `python scripts/check_mediapipe.py` | `scripts/check_mediapipe.py` | Verify MediaPipe installation |
| `python scripts/auto_config.py` | `scripts/auto_config.py` | Auto-detect hardware and generate optimized config |
| `python -m src.data.preprocess` | `src/data/preprocess.py` | Extract keypoints from videos, create splits |
| `python -m src.training.train` | `src/training/train.py` | Train a model |
| `python -m src.training.evaluate` | `src/training/evaluate.py` | Evaluate a trained model |
| `python -m src.inference.predict` | `src/inference/predict.py` | Predict on a single video |
| `python -m src.inference.live_demo` | `src/inference/live_demo.py` | Run real-time webcam demo |
| `python -m src.inference.export_onnx` | `src/inference/export_onnx.py` | Export model to ONNX format |
