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
                    │   (episodic    │              │
                    │    prototypical│              v
                    │    training)   │        eval_results/
                    │               │        confusion_matrix.png
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
            │ train_   │  │evaluate. │  │ predict.  │  │ live_demo. │
            │ proto-   │  │  py      │  │   py      │  │    py      │
            │ typical  │  └──┬───┬──┘  └──┬──┬─────┘  └──┬──┬─────┘
            └──┬───┬───┘     │   │        │  │           │  │
               │   │    ┌────┘   │        │  │           │  │
          ┌────┘   │    v        v        │  │           │  │
          v        v                      │  │           │  │
    ┌──────────┐ ┌──────────┐             │  │           │  │
    │augment.py│ │dataset.py│             │  │           │  │
    └──────────┘ └──────────┘             │  │           │  │
                                          │  │           │  │
    ┌──────────────────┐                  │  │           │  │
    │episode_sampler.py│ <── train_proto  │  │           │  │
    └──────────────────┘                  │  │           │  │
                                          │  │           │  │
               ┌──────────────────────────┘  │           │  │
               v                             v           v  │
    ┌────────────────┐            ┌─────────────────┐      │
    │ preprocess.py  │            │ prototypical.py │      │
    │ (normalize,    │            │ (Prototypical   │<─────┘
    │  keypoints)    │            │  Network,       │
    └────────────────┘            │  build_model)   │
                                  └───────┬─────────┘
                                          │
                                          v
                                  ┌─────────────────┐
                                  │   stgcn.py      │
                                  │ (STGCNEncoder,  │
                                  │  graph utils)   │
                                  └─────────────────┘
```

---

## Module Details

### Scripts (Entry Points)

| Script | Purpose | Imports From | Output |
|--------|---------|--------------|--------|
| `scripts/download_wlasl.py` | Download annotations + print video instructions | `src.data.preprocess` | `data/annotations/`, `data/raw/` |
| `scripts/download_kaggle.py` | Download full video archive from Kaggle | `src.data.preprocess` | `data/raw/*.mp4` |
| `scripts/validate_videos.py` | Detect and remove fake HTML video files | (none) | Cleaned `data/raw/` |
| `scripts/reset_configs.py` | Reset YAML config to README defaults | (none) | `configs/stgcn_proto.yaml` |
| `scripts/check_mediapipe.py` | Verify MediaPipe installation, diagnose `solutions` import issues | (none) | Diagnostic output to stdout |
| `scripts/auto_config.py` | Auto-detect hardware (CUDA/MPS/CPU) and generate optimized config | (none) | `configs/stgcn_proto.yaml` |

### Data Pipeline (`src/data/`)

| Module | Key Functions / Classes | Used By |
|--------|------------------------|---------|
| `preprocess.py` | `download_wlasl_annotations()`, `parse_wlasl_annotations()`, `extract_keypoints_mediapipe()`, `normalize_keypoints()`, `preprocess_dataset()`, `create_splits()` | `download_wlasl.py`, `download_kaggle.py`, `predict.py`, `live_demo.py` |
| `augment.py` | `TemporalCrop`, `TemporalSpeedPerturb`, `KeypointHorizontalFlip`, `KeypointYawRotation`, `KeypointRotation`, `KeypointTranslation`, `KeypointDropout`, `KeypointNoise`, `KeypointScale`, `Compose`, `get_train_transforms()`, `get_val_transforms()` | `train_prototypical.py`, `evaluate.py`, `predict.py` |
| `dataset.py` | `WLASLKeypointDataset`, `get_dataloader()` | `train_prototypical.py`, `evaluate.py`, `predict.py` |
| `episode_sampler.py` | `EpisodicBatchSampler` | `train_prototypical.py` |

### Models (`src/models/`)

| Module | Key Classes | Build Function |
|--------|-------------|----------------|
| `stgcn.py` | `SpatialGraphConv`, `STGCNBlock`, `STGCNBranch`, `STGCNEncoder` | `build_stgcn_encoder(cfg)` |
| `prototypical.py` | `PrototypicalNetwork` | `build_model(cfg)` |

`build_model()` creates an `STGCNEncoder` wrapped in a `PrototypicalNetwork`. The encoder processes body (33 joints), left hand (21 joints), and right hand (21 joints) through separate ST-GCN branches, then merges into a single L2-normalized embedding.

### Training (`src/training/`)

| Module | Key Functions | Imports From |
|--------|---------------|--------------|
| `config.py` | `Config` (dataclass), `load_config()`, `save_config()` | (none — leaf dependency) |
| `train.py` | CLI dispatcher | `config`, `train_prototypical` |
| `train_prototypical.py` | `_split_episode()`, `train_one_epoch()`, `validate()`, `main()` | `config`, `augment`, `dataset`, `episode_sampler`, `prototypical` |
| `evaluate.py` | `compute_metrics()`, `plot_confusion_matrix()`, `find_hard_negatives()`, `evaluate_latency()`, `main()` | `config`, `augment`, `dataset`, `prototypical` |

### Inference (`src/inference/`)

| Module | Key Classes / Functions | Imports From |
|--------|------------------------|--------------|
| `predict.py` | `SignPredictor`, `_load_class_names()` | `config`, `augment`, `preprocess`, `prototypical`, `dataset` |
| `live_demo.py` | `FrameBuffer`, `LivePredictor`, `ASLDisplay`, `run_demo()` | `config`, `preprocess`, `prototypical` |
| `export_onnx.py` | `export_to_onnx()`, `verify_onnx()`, `benchmark_onnx()` | `config`, `prototypical` |

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
  │    1. shoulder-center + scale   │
  │    2. face-center relative      │
  │    3. depth normalization       │
  │    4. hand-relative to wrist    │
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
  │  train_prototypical.py          │
  │  Episodic training:             │
  │    EpisodicBatchSampler         │
  │      N-way K-shot + Q-query     │
  │    _split_episode():            │
  │      split into support/query   │
  │    model.forward():             │
  │      encode → prototypes →      │
  │      negative distances         │
  │    F.cross_entropy(logits, y)   │
  │  validate():                    │
  │    compute prototypes from      │
  │    full training set            │
  │    classify val by nearest      │  ──> checkpoints/best_model.pt
  │    prototype                    │  ──> logs/ (TensorBoard)
  └─────────────────────────────────┘
```

### Inference Data Flow

```
Single Video (predict.py):

  video.mp4 ──> MediaPipe ──> normalize ──> velocity ──> encoder ──> prototype distances ──> top-5
       │                                        ^
       │         OR                             │
  keypoints.npy ──────────> velocity ───────────┘


Live Demo (live_demo.py):

  Webcam ──> FrameBuffer(T=64) ──> MediaPipe ──> normalize ──> velocity ──> model.classify()
    │              │                                                            │
    │         rolls every                                                       v
    │         0.5 seconds                                                  predictions
    │                                                                           │
    v                                                                           v
  Display <───── overlay predicted gloss + confidence (smoothed over 5 windows)


ONNX Export (export_onnx.py):

  checkpoint ──> load encoder ──> torch.onnx.export() ──> encoder.onnx (produces embeddings)
                                       │                  prototypes stored separately
                              verify (optional) ──> ONNX Runtime forward pass
                              benchmark (optional) ──> avg latency over 100 runs
```

### Model Architecture Flow (ST-GCN + Prototypical Network)

```
Input: (batch, T, 543*C)     C = 3 (xyz) or 6 (xyz + velocity)
              │
              v
    ┌──────────────────────────┐
    │  Partition keypoints     │
    │  Body: joints 0-32 (33) │
    │  LHand: joints 33-53    │
    │  RHand: joints 54-74    │
    └────┬─────┬─────┬────────┘
         │     │     │
         v     v     v
    ┌────────┐ ┌────────┐ ┌────────┐
    │ Body   │ │ LHand  │ │ RHand  │   Each branch: stack of STGCNBlocks
    │ Branch │ │ Branch │ │ Branch │     Spatial GCN (bone connections)
    │ (33 V) │ │ (21 V) │ │ (21 V) │     + Temporal Conv (kernel=9)
    └───┬────┘ └───┬────┘ └───┬────┘     + Residual + BatchNorm
        │          │          │
        v          v          v
    ┌──────────────────────────┐
    │  Concatenate branch      │  3 × branch_out_channels
    │  outputs                 │
    └─────────┬────────────────┘
              v
    ┌─────────────────────┐
    │  Linear projection  │  → embedding_dim (d_model)
    │  + LayerNorm        │
    └─────────┬───────────┘
              v
    ┌─────────────────────┐
    │  L2 normalize       │  unit-length embedding
    └─────────┬───────────┘
              v
    Output: (batch, embedding_dim)

Training (episodic):
    support embeddings ──> prototypes (mean per class)
    query embeddings ──> distances to prototypes ──> cross-entropy loss

Inference:
    prototypes computed from full training set (stored in model)
    new sample ──> encoder ──> distance to all prototypes ──> predicted class
```

---

## Configuration Flow

```
configs/stgcn_proto.yaml ──> load_config() ──> Config dataclass
                                                       │
                                           ┌───────────┼───────────┐
                                           v           v           v
                                     train.py    evaluate.py  predict.py
                                                               live_demo.py
                                                               export_onnx.py

Config.__post_init__() auto-derives:
    wlasl_variant: 100  ──>  num_classes: 100,  gcn_channels: [64,128,128],   d_model: 128, dropout: 0.1
    wlasl_variant: 300  ──>  num_classes: 300,  gcn_channels: [64,128,256],   d_model: 192, dropout: 0.15
    wlasl_variant: 1000 ──>  num_classes: 1000, gcn_channels: [64,128,256],   d_model: 256, dropout: 0.2
    wlasl_variant: 2000 ──>  num_classes: 2000, gcn_channels: [64,128,256,256], d_model: 384, dropout: 0.2
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
| `test_models.py` | `src/models/` — STGCNEncoder, PrototypicalNetwork, graph construction |
| `test_predict.py` | `src/inference/predict.py` — SignPredictor inference paths |
| `test_preprocess.py` | `src/data/preprocess.py` — normalization, annotation parsing, splits |
| `test_train.py` | `src/training/train_prototypical.py` — episode splitting |
| `test_auto_config.py` | `scripts/auto_config.py` — hardware detection, config generation |
| `test_dependencies.py` | All `requirements.txt` libraries — version checks, feature compatibility, src module imports |

All tests use `conftest.py` shared fixtures (tmp datasets, keypoint generators) and are fully isolated (no project data or checkpoints needed).

---

## CLI Entry Points

| Command | Module | What It Does |
|---------|--------|-------------|
| `python scripts/download_wlasl.py` | `scripts/download_wlasl.py` | Download annotations, print video download instructions |
| `python scripts/download_kaggle.py` | `scripts/download_kaggle.py` | Download all videos from Kaggle |
| `python scripts/validate_videos.py` | `scripts/validate_videos.py` | Scan and clean fake video files |
| `python scripts/reset_configs.py` | `scripts/reset_configs.py` | Reset config to recommended defaults |
| `python scripts/check_mediapipe.py` | `scripts/check_mediapipe.py` | Verify MediaPipe installation |
| `python scripts/auto_config.py` | `scripts/auto_config.py` | Auto-detect hardware and generate optimized config |
| `python -m src.data.preprocess` | `src/data/preprocess.py` | Extract keypoints from videos, create splits |
| `python -m src.training.train` | `src/training/train.py` | Train the model (dispatches to prototypical training) |
| `python -m src.training.evaluate` | `src/training/evaluate.py` | Evaluate a trained model |
| `python -m src.inference.predict` | `src/inference/predict.py` | Predict on a single video |
| `python -m src.inference.live_demo` | `src/inference/live_demo.py` | Run real-time webcam demo |
| `python -m src.inference.export_onnx` | `src/inference/export_onnx.py` | Export encoder to ONNX format |
