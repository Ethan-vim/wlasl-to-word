# Live American Sign Language Recognition using WLASL

## Project Prompt — Deep Learning / Machine Learning

---

## 1. Project Overview

Build a real-time American Sign Language (ASL) word-level recognition system that captures live webcam video, processes hand/body movements frame-by-frame, and predicts the signed word using a deep learning model trained on the **WLASL (Word-Level American Sign Language)** dataset.

The final deliverable is a working application that opens a webcam feed, detects when a user is signing, runs inference on the captured clip, and displays the predicted English word on screen with a confidence score.

---

## 2. Dataset

### 2.1 Source

- **Name:** WLASL (Word-Level American Sign Language)
- **Paper:** *"Word-level Deep Sign Language Recognition from Video" (WACV 2020)* by Dongxu Li et al.
- **Repository:** `https://github.com/dxli94/WLASL`
- **License:** Check the repository for current licensing terms before use.

### 2.2 Dataset Variants

| Variant     | Glosses (Words) | Video Samples | Notes                          |
|-------------|-----------------|---------------|--------------------------------|
| WLASL-100   | 100             | ~2,000        | Good starting point            |
| WLASL-300   | 300             | ~5,000        | Balanced difficulty             |
| WLASL-1000  | 1,000           | ~13,000       | Medium scale                   |
| WLASL-2000  | 2,000           | ~21,000       | Full dataset, most challenging |

### 2.3 Data Characteristics

- Videos are sourced from ASL educational websites and dictionaries.
- Each video contains a single signer performing a single isolated word/gloss.
- Signers vary in appearance, background, lighting, and signing speed.
- Videos are variable length (typically 0.5–3 seconds).
- Annotation format: JSON mapping video IDs to gloss labels, bounding boxes, and signer metadata.

### 2.4 Recommended Starting Point

Begin with **WLASL-100** for prototyping and validation, then scale to WLASL-300 or beyond once the pipeline is stable.

---

## 3. Problem Definition

### 3.1 Task Type

**Isolated Word-Level Sign Language Recognition (ISLR)**
— Given a short video clip of a person signing, classify which word (gloss) is being performed.

### 3.2 Input

- A variable-length video clip (sequence of RGB frames) or a sequence of extracted pose/hand keypoints.

### 3.3 Output

- A predicted gloss (English word) with a confidence score.
- Top-K predictions with probabilities (for display or fallback logic).

### 3.4 Evaluation Metrics

- **Top-1 Accuracy** — Primary metric. The predicted word exactly matches the ground truth.
- **Top-5 Accuracy** — Secondary metric. The correct word appears in the top 5 predictions.
- **Per-class Accuracy** — To identify weak/confused classes.
- **Inference Latency** — Time from clip capture to prediction (target: under 500ms for real-time use).

---

## 4. Technical Architecture

### 4.1 High-Level Pipeline

```
┌──────────────┐    ┌────────────────┐    ┌─────────────────┐    ┌────────────────┐
│  Webcam Feed │───▶│  Preprocessing │───▶│  Feature Extract │───▶│  Classifier    │
│  (OpenCV)    │    │  & Keypoint    │    │  (Backbone)      │    │  (Temporal)    │
│              │    │  Extraction    │    │                  │    │                │
└──────────────┘    └────────────────┘    └─────────────────┘    └────────────────┘
                                                                        │
                                                                        ▼
                                                                 ┌──────────────┐
                                                                 │  Predicted   │
                                                                 │  Word + Conf │
                                                                 └──────────────┘
```

### 4.2 Approach Options (Choose One or Combine)

#### Approach A — Pose/Keypoint-Based (Recommended for Beginners)

1. Use **MediaPipe Holistic** to extract per-frame keypoints: 33 pose landmarks, 21 per-hand landmarks (x2), and 468 face landmarks.
2. Flatten or structure the keypoints into a feature vector per frame.
3. Stack frames into a fixed-length temporal sequence (e.g., 30–64 frames via padding/sampling).
4. Feed the sequence into a temporal model (LSTM, GRU, Transformer, or 1D-CNN).

**Pros:** Lightweight, fast inference, abstracts away background/appearance.
**Cons:** Loses fine-grained visual detail (texture, finger occlusion cues).

#### Approach B — Video-Based (RGB Frames)

1. Sample or resize each video to a fixed number of frames (e.g., 16, 32, or 64).
2. Resize frames to a consistent spatial resolution (e.g., 224×224).
3. Use a spatiotemporal backbone:
   - **I3D** (Inflated 3D ConvNet) — strong baseline for video classification.
   - **R(2+1)D** — factorized 3D convolutions, efficient.
   - **SlowFast** — dual-pathway for capturing both slow and fast motion.
   - **Video Swin Transformer** or **TimeSformer** — attention-based, state-of-the-art.
4. Add a classification head (FC layers) on top of the backbone features.

**Pros:** Captures full visual context.
**Cons:** Heavier compute, needs GPU for real-time inference, more prone to background bias.

#### Approach C — Hybrid (Pose + RGB Fusion)

1. Extract both RGB features (via a 2D CNN per frame or 3D CNN) and keypoint sequences.
2. Fuse via late fusion (concatenate feature vectors before classifier), or cross-attention.
3. Classify using a shared temporal head.

**Pros:** Best accuracy potential.
**Cons:** Most complex to implement and tune.

---

## 5. Detailed Implementation Plan

### Phase 1 — Data Preparation

1. **Download the WLASL dataset** using the provided scripts in the official repo. Handle missing/broken video links gracefully (some URLs expire over time).
2. **Parse annotations** from the JSON file. Map each video to its gloss label, split (train/val/test), and signer ID.
3. **Extract frames** from each video at a consistent FPS (e.g., 25 fps). Store as image sequences or keep as video files.
4. **Extract keypoints (if using Approach A):**
   - Run MediaPipe Holistic on every frame of every video.
   - Save keypoint arrays as `.npy` files: shape `(num_frames, num_keypoints, 3)`.
   - Handle detection failures (missing hand, occluded frames) with zero-padding or interpolation.
5. **Normalize and augment:**
   - Normalize keypoint coordinates relative to a reference point (e.g., nose or shoulder center).
   - Scale to a unit bounding box.
   - Temporal augmentation: random speed perturbation (±15%), temporal crop/shift.
   - Spatial augmentation (for RGB): random horizontal flip (with caution — some signs are asymmetric), color jitter, random crop.
6. **Create a fixed-length representation:**
   - Uniformly sample or interpolate each clip to a fixed frame count `T` (e.g., 32 or 64).
   - Pad shorter clips with zeros or repeat the last frame.

### Phase 2 — Model Development

1. **Define the model architecture:**

   **For Approach A (Keypoint-based):**
   ```
   Input: (batch, T, num_keypoints * 3)
       → Linear projection (embed dim 128–256)
       → Positional encoding
       → N × Transformer encoder layers (or 2-layer BiLSTM)
       → Global average pooling over time
       → Dropout (0.3–0.5)
       → FC → num_classes
   ```

   **For Approach B (Video-based):**
   ```
   Input: (batch, 3, T, 224, 224)
       → Pretrained I3D / R(2+1)D / SlowFast backbone (freeze early layers initially)
       → Global spatiotemporal pooling
       → Dropout (0.3–0.5)
       → FC → num_classes
   ```

2. **Loss function:** Cross-Entropy Loss. Consider Label Smoothing (0.1) to improve generalization.
3. **Optimizer:** AdamW with weight decay 1e-4.
4. **Learning rate schedule:** Cosine annealing with warm-up (5–10 epochs warm-up, total 50–100 epochs).
5. **Batch size:** 16–32 (adjust based on GPU memory).
6. **Mixed precision training (FP16):** Use PyTorch AMP for faster training and lower memory usage.

### Phase 3 — Training & Validation

1. Train on the official train split, validate on the val split.
2. Monitor top-1 and top-5 accuracy on validation at each epoch.
3. Save the best model checkpoint based on val top-1 accuracy.
4. Run a confusion matrix analysis on the validation set to find commonly confused sign pairs.
5. Experiment log: track all hyperparameters, metrics, and notes per run using **Weights & Biases**, **MLflow**, or a simple CSV log.

### Phase 4 — Live Inference Pipeline

1. **Webcam capture:** Use OpenCV (`cv2.VideoCapture(0)`) to read frames in real time.
2. **Buffering strategy:**
   - Maintain a rolling buffer of the last `T` frames (e.g., 64 frames ≈ 2.5 seconds at 25fps).
   - Run inference on the buffer at a fixed interval (e.g., every 0.5 seconds) or on a trigger (e.g., hand detected in frame).
3. **Preprocessing in real time:**
   - Run MediaPipe on each incoming frame (for Approach A).
   - Resize/normalize frames (for Approach B).
4. **Inference:**
   - Convert the buffer into a tensor, send to the model.
   - Apply softmax to get class probabilities.
   - Display the top-1 prediction and confidence on the video feed using `cv2.putText`.
5. **Smoothing / Debouncing:**
   - Average predictions over the last N inference windows to reduce flickering.
   - Only display a prediction if confidence exceeds a threshold (e.g., 0.6).
   - Optionally require the same prediction for K consecutive windows before displaying.
6. **Display:**
   - Overlay the predicted word, confidence bar, and optionally the top-5 predictions on the OpenCV window.
   - Draw the detected hand/pose skeleton on the feed for visual feedback.

### Phase 5 — Optimization & Deployment

1. **Model optimization for speed:**
   - Export to ONNX and run with ONNX Runtime for faster CPU/GPU inference.
   - Quantize the model (INT8) using PyTorch quantization or TensorRT.
   - Profile inference latency and target under 200ms per prediction.
2. **Optional UI:**
   - Wrap the application in a simple GUI using Tkinter, PyQt, or a web frontend (Gradio / Streamlit).
   - Add a vocabulary panel showing all recognized signs.
   - Add a "sentence builder" that concatenates recognized words over time.
3. **Edge deployment (stretch goal):**
   - Convert to TFLite or CoreML for mobile deployment.
   - Test on a Raspberry Pi or Jetson Nano for embedded use.

---

## 6. Suggested Technology Stack

| Component             | Recommended Tools                                    |
|-----------------------|------------------------------------------------------|
| Language              | Python 3.9+                                          |
| Deep Learning         | PyTorch (preferred) or TensorFlow/Keras              |
| Video I/O             | OpenCV, Decord (for fast video loading)              |
| Keypoint Extraction   | MediaPipe Holistic, or OpenPose                      |
| Data Handling         | NumPy, pandas, scikit-learn (for splits/metrics)     |
| Augmentation          | Albumentations (spatial), custom temporal transforms |
| Experiment Tracking   | Weights & Biases, MLflow, or TensorBoard             |
| Model Export          | ONNX, TorchScript                                    |
| Fast Inference        | ONNX Runtime, TensorRT                               |
| UI (optional)         | Gradio, Streamlit, or Tkinter                        |
| Version Control       | Git + Git LFS (for model checkpoints)                |

---

## 7. Expected Baseline Results

Based on published benchmarks on WLASL:

| Model Type             | WLASL-100 Top-1 | WLASL-300 Top-1 | WLASL-2000 Top-1 |
|------------------------|------------------|------------------|-------------------|
| Pose-based LSTM        | ~50–60%          | ~40–50%          | ~25–35%           |
| Pose-based Transformer | ~60–70%          | ~50–58%          | ~35–45%           |
| I3D (RGB)              | ~65–75%          | ~55–63%          | ~40–50%           |
| I3D + Pose Fusion      | ~70–78%          | ~58–66%          | ~45–53%           |

These are approximate ranges. Your results will depend on how many videos are still downloadable, augmentation strategy, and hyperparameter tuning.

---

## 8. Known Challenges & Mitigations

| Challenge | Mitigation |
|---|---|
| **Missing videos** — Many original URLs are dead. | Use the community-maintained mirrors. Pre-filter the annotation file to only include videos you have. Re-balance classes after filtering. |
| **Class imbalance** — Some glosses have far fewer samples. | Use weighted sampling, focal loss, or oversample minority classes with augmentation. |
| **Signer variability** — Different people sign at different speeds and scales. | Normalize keypoints spatially. Use temporal augmentation. Include signer-aware splits to avoid data leakage. |
| **Background bias** — RGB models may memorize backgrounds instead of signs. | Crop to the signer's bounding box. Use keypoint-based approach. Apply background augmentation. |
| **Ambiguous/similar signs** — Some words differ by very subtle hand shapes. | Focus augmentation and hard-negative mining on confused pairs. Consider hierarchical classification (group similar signs). |
| **Real-time latency** — Full video models are slow. | Use a lightweight backbone (MobileNet-based), quantize, or use keypoint approach. Profile and optimize the bottleneck. |

---

## 9. Stretch Goals

- **Continuous sign recognition** — Move beyond isolated word recognition to handle continuous signing (sequence-to-sequence with CTC or attention decoder).
- **Fingerspelling module** — Add a separate classifier for the ASL alphabet to handle spelled-out words.
- **Two-hand interaction modeling** — Explicitly model the spatial relationship between left and right hands using graph neural networks.
- **Self-supervised pretraining** — Pretrain the feature extractor on unlabeled sign language video (e.g., from YouTube ASL channels) using contrastive learning before fine-tuning on WLASL.
- **Multi-modal output** — Display the recognized word as text, speak it aloud via text-to-speech, or translate to another language.
- **User enrollment** — Allow a user to record their own signs for new words and fine-tune the model on the fly (few-shot adaptation).

---

## 10. Project Directory Structure

```
wlasl-live-recognition/
├── data/
│   ├── raw/                    # Raw downloaded videos
│   ├── processed/              # Extracted frames or keypoints (.npy)
│   ├── annotations/            # WLASL JSON annotation files
│   └── splits/                 # Train/val/test split CSVs
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   ├── preprocess.py       # Frame extraction, keypoint extraction
│   │   └── augment.py          # Temporal and spatial augmentations
│   ├── models/
│   │   ├── pose_transformer.py # Keypoint-based Transformer model
│   │   ├── video_i3d.py        # RGB video model (I3D or similar)
│   │   └── fusion.py           # Multi-modal fusion model
│   ├── training/
│   │   ├── train.py            # Training loop
│   │   ├── evaluate.py         # Evaluation and metrics
│   │   └── config.py           # Hyperparameters and paths
│   └── inference/
│       ├── live_demo.py        # Webcam + real-time inference
│       ├── predict.py          # Single-video prediction
│       └── export_onnx.py      # Model export for deployment
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_keypoint_visualization.ipynb
│   └── 03_error_analysis.ipynb
├── checkpoints/                # Saved model weights
├── logs/                       # Training logs
├── requirements.txt
└── README.md
```
Note: README.md should include how to use this project with a full tutorial on how to setup and how to use

---

## 11. References

1. Dongxu Li et al., "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison," WACV 2020.
2. Joao Carreira & Andrew Zisserman, "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset," CVPR 2017 (I3D).
3. Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines," 2019.
4. Arnab et al., "ViViT: A Video Vision Transformer," ICCV 2021.
5. Feichtenhofer et al., "SlowFast Networks for Video Recognition," ICCV 2019.

---

*This prompt is designed to be handed to a developer or ML practitioner as a self-contained project specification. Adjust the scope (WLASL variant, model complexity, deployment target) based on available compute, timeline, and team experience.*