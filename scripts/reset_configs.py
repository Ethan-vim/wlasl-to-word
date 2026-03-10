"""
Reset all YAML configs in configs/ to the recommended defaults from README.md.

This script overwrites:
    - configs/pose_transformer.yaml  (Approach A — WLASL100 recommended starting point)
    - configs/video_classifier.yaml  (Approach B — Video Classifier)
    - configs/fusion.yaml            (Approach C — Hybrid Fusion)

Usage:
    python scripts/reset_configs.py
    python scripts/reset_configs.py --dry-run       # preview without writing
    python scripts/reset_configs.py --only pose      # reset only pose_transformer.yaml
    python scripts/reset_configs.py --only video     # reset only video_classifier.yaml
    python scripts/reset_configs.py --only fusion    # reset only fusion.yaml
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ---------------------------------------------------------------------------
# Default configs — these match the README.md "Recommended Configurations"
# section combined with Config dataclass defaults from src/training/config.py.
# ---------------------------------------------------------------------------

POSE_TRANSFORMER_YAML = """\
## Approach A: Pose/Keypoint Transformer
## Optimized configuration (WLASL100).
## Note: d_model, nhead, num_layers, dropout are auto-scaled per variant
## by Config.__post_init__ — values here match the WLASL100 defaults.

approach: pose_transformer
wlasl_variant: 100
# num_classes is auto-derived from wlasl_variant (100 -> 100, 300 -> 300, etc.)
num_keypoints: 543

# Temporal
T: 64

# Features
use_motion: true  # Concatenate velocity (frame differences) with position

# Model architecture (auto-scaled by __post_init__ for the variant)
d_model: 128
nhead: 4
num_layers: 2
dropout: 0.1

# Data loading
num_workers: 4              # parallel data-loading workers (0 = main process only)

# Training
epochs: 250
batch_size: 32
lr: 3.0e-4
weight_decay: 5.0e-4
warmup_epochs: 15
label_smoothing: 0.0
grad_clip: 1.0
fp16: true
weighted_sampling: true  # important — classes are imbalanced
early_stopping_patience: 50
mixup_alpha: 0.15  # Light mixup for diversity without flooring loss

# Scheduler
scheduler: cosine

# Evaluation
use_tta: false  # Test-time augmentation (horizontal flip averaging)

# Logging
use_wandb: false
use_tensorboard: true
log_interval: 10

# Inference
confidence_threshold: 0.6
smoothing_window: 5
buffer_size: 64
fps_display: true

# Paths
data_dir: data
output_dir: outputs
checkpoint_dir: checkpoints
log_dir: logs
"""

VIDEO_CLASSIFIER_YAML = """\
## Approach B: RGB Video Classifier
## Uses pretrained 3D CNN backbones (R(2+1)D, R3D, SlowFast, etc.)
## Optimized configuration (WLASL100).

approach: video
backbone: r2plus1d_18
pretrained: true
wlasl_variant: 100
# num_classes is auto-derived from wlasl_variant (100 -> 100, 300 -> 300, etc.)

# Temporal & spatial
T: 64                     # higher temporal resolution for better sign recognition
image_size: 224            # reduce to 112 if GPU memory is tight

# Model
dropout: 0.3               # video backbones need moderate dropout (not auto-scaled)

# Data loading
num_workers: 4              # parallel data-loading workers (0 = main process only)

# Training
epochs: 250
batch_size: 8              # 3D CNNs need small batches
lr: 1.0e-4                 # lower LR for finetuning pretrained backbone
weight_decay: 5.0e-4
warmup_epochs: 10
label_smoothing: 0.0
grad_clip: 1.0
fp16: true                 # essential for video models
weighted_sampling: false
early_stopping_patience: 40

# Scheduler
scheduler: cosine

# Logging
use_wandb: false
use_tensorboard: true
log_interval: 10

# Inference
confidence_threshold: 0.6
smoothing_window: 5
buffer_size: 64
fps_display: true

# Paths
data_dir: data
output_dir: outputs
checkpoint_dir: checkpoints
log_dir: logs
"""

FUSION_YAML = """\
## Approach C: Hybrid Fusion (Pose + Video)
## Combines Approach A and B for best accuracy.
## Optimized configuration (WLASL100).

approach: fusion
fusion: concat             # start with concat, try attention if concat plateaus
fusion_dim: 256

# Sub-model configs
backbone: r2plus1d_18
pretrained: true
num_keypoints: 543

# Dataset
wlasl_variant: 100
# num_classes is auto-derived from wlasl_variant (100 -> 100, 300 -> 300, etc.)
T: 64
image_size: 224

# Pose Transformer settings (auto-scaled by __post_init__ for the variant)
d_model: 128
nhead: 4
num_layers: 2
dropout: 0.1

# Data loading
num_workers: 4              # parallel data-loading workers (0 = main process only)

# Training
epochs: 250
batch_size: 8
lr: 1.0e-4
weight_decay: 5.0e-4
warmup_epochs: 10
label_smoothing: 0.0
grad_clip: 1.0
fp16: true
weighted_sampling: false
early_stopping_patience: 40
mixup_alpha: 0.1            # very light mixup

# Scheduler
scheduler: cosine

# Logging
use_wandb: false
use_tensorboard: true
log_interval: 10

# Inference
confidence_threshold: 0.6
smoothing_window: 5
buffer_size: 64
fps_display: true

# Paths
data_dir: data
output_dir: outputs
checkpoint_dir: checkpoints
log_dir: logs
"""

CONFIGS = {
    "pose": ("pose_transformer.yaml", POSE_TRANSFORMER_YAML),
    "video": ("video_classifier.yaml", VIDEO_CLASSIFIER_YAML),
    "fusion": ("fusion.yaml", FUSION_YAML),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset configs/ to README.md recommended defaults"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying files",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=list(CONFIGS),
        help="Reset only one config file (pose, video, or fusion)",
    )
    args = parser.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    targets = {args.only: CONFIGS[args.only]} if args.only else CONFIGS

    for key, (filename, content) in targets.items():
        path = CONFIGS_DIR / filename

        if args.dry_run:
            print(f"[DRY RUN] Would write {path}")
            print(content)
            print("=" * 60)
            continue

        path.write_text(content, encoding="utf-8")
        print(f"  Reset {path}")

    if not args.dry_run:
        print(f"\n  {len(targets)} config(s) reset to README.md recommended defaults.")


if __name__ == "__main__":
    main()
