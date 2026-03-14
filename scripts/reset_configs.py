"""
Reset the YAML config in configs/ to the recommended defaults.

This script overwrites:
    - configs/stgcn_proto.yaml  (ST-GCN + Prototypical Network)

Usage:
    python scripts/reset_configs.py
    python scripts/reset_configs.py --dry-run       # preview without writing
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ---------------------------------------------------------------------------
# Default config — matches the README.md "Recommended Configuration"
# section combined with Config dataclass defaults from src/training/config.py.
# ---------------------------------------------------------------------------

STGCN_PROTO_YAML = """\
## ST-GCN + Prototypical Network
## Optimized configuration (WLASL100).
## Note: gcn_channels, d_model, dropout are auto-scaled per variant
## by Config.__post_init__ — values here match the WLASL100 defaults.
## For device-specific tuning, run: python scripts/auto_config.py

approach: stgcn_proto
wlasl_variant: 100
# num_classes is auto-derived from wlasl_variant (100 -> 100, 300 -> 300, etc.)
num_keypoints: 543

# Temporal
T: 64

# Features
use_motion: true  # Concatenate velocity (frame differences) with position

# Model (ST-GCN)
d_model: 128
gcn_channels: [64, 128, 128]
num_layers: 3
dropout: 0.1

# Prototypical training
# MPS users: use n_way: 10 and num_episodes: 200 (auto_config.py does this)
n_way: 20
k_shot: 3
q_query: 2
num_episodes: 500

# Data loading
# MPS users: use num_workers: 0 (multiprocessing deadlocks on MPS)
num_workers: 4              # parallel data-loading workers (0 = main process only)

# Training
epochs: 200
batch_size: 32
lr: 1.0e-3
weight_decay: 1.0e-4
warmup_epochs: 10
grad_clip: 1.0
fp16: false                 # only enable on CUDA — MPS does not support FP16/AMP
weighted_sampling: false
early_stopping_patience: 30

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

CONFIGS = {
    "stgcn_proto": ("stgcn_proto.yaml", STGCN_PROTO_YAML),
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
    args = parser.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    for key, (filename, content) in CONFIGS.items():
        path = CONFIGS_DIR / filename

        if args.dry_run:
            print(f"[DRY RUN] Would write {path}")
            print(content)
            print("=" * 60)
            continue

        path.write_text(content, encoding="utf-8")
        print(f"  Reset {path}")

    if not args.dry_run:
        print(f"\n  {len(CONFIGS)} config(s) reset to README.md recommended defaults.")


if __name__ == "__main__":
    main()
