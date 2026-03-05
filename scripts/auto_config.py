"""
Auto-detect hardware and generate an optimized YAML config.

Probes your GPU (CUDA), Apple Silicon (MPS), or CPU, determines a
performance tier, and writes a ready-to-train config for the selected
approach and WLASL variant.

Usage:
    python scripts/auto_config.py --approach pose
    python scripts/auto_config.py --approach video --variant 100
    python scripts/auto_config.py --approach fusion --variant 300
    python scripts/auto_config.py --approach pose --dry-run
    python scripts/auto_config.py --approach pose --device cpu
    python scripts/auto_config.py --approach pose --backup
"""

import argparse
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

VALID_APPROACHES = ("pose", "video", "fusion")
VALID_VARIANTS = (100, 300, 1000, 2000)

APPROACH_TO_FILE = {
    "pose": "pose_transformer.yaml",
    "video": "video_classifier.yaml",
    "fusion": "fusion.yaml",
}

APPROACH_TO_NAME = {
    "pose": "pose_transformer",
    "video": "video",
    "fusion": "fusion",
}


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


@dataclass
class HardwareInfo:
    """Detected hardware information."""

    device: str  # "cuda", "mps", "cpu"
    device_name: str  # e.g. "NVIDIA RTX 4090", "Apple M2", "CPU"
    vram_gb: float  # 0.0 for CPU/MPS
    cuda_version: str  # e.g. "12.1", "" for non-CUDA
    cpu_cores: int
    platform_name: str
    torch_version: str
    gpu_count: int  # number of CUDA GPUs


def detect_hardware(device_override: str | None = None) -> HardwareInfo:
    """Auto-detect the available hardware.

    Parameters
    ----------
    device_override : str or None
        Force a specific device ("cuda", "mps", or "cpu").

    Returns
    -------
    HardwareInfo
    """
    cpu_cores = os.cpu_count() or 1
    plat = platform.platform()

    try:
        import torch
    except ImportError:
        return HardwareInfo(
            device="cpu",
            device_name=platform.processor() or "Unknown CPU",
            vram_gb=0.0,
            cuda_version="",
            cpu_cores=cpu_cores,
            platform_name=plat,
            torch_version="(not installed)",
            gpu_count=0,
        )

    torch_ver = torch.__version__

    # Determine device
    if device_override:
        device = device_override
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Gather device-specific info
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        vram_gb = props.total_mem / (1024**3)
        cuda_version = torch.version.cuda or ""
        gpu_count = torch.cuda.device_count()
    elif device == "mps":
        proc = platform.processor()
        device_name = f"Apple {proc}" if proc else "Apple Silicon"
        vram_gb = 0.0
        cuda_version = ""
        gpu_count = 0
    else:
        device_name = platform.processor() or "CPU"
        vram_gb = 0.0
        cuda_version = ""
        gpu_count = 0

    return HardwareInfo(
        device=device,
        device_name=device_name,
        vram_gb=round(vram_gb, 1),
        cuda_version=cuda_version,
        cpu_cores=cpu_cores,
        platform_name=plat,
        torch_version=torch_ver,
        gpu_count=gpu_count,
    )


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


def determine_tier(hw: HardwareInfo) -> str:
    """Classify hardware into a performance tier.

    Returns
    -------
    str
        One of "high", "mid", "low", "cpu".
    """
    if hw.device == "cuda":
        if hw.vram_gb >= 16:
            return "high"
        elif hw.vram_gb >= 8:
            return "mid"
        else:
            return "low"
    return "cpu"  # MPS and CPU both get cpu tier


# ---------------------------------------------------------------------------
# Config value generation
# ---------------------------------------------------------------------------


def build_config_values(
    approach: str,
    variant: int,
    tier: str,
    hw: HardwareInfo,
) -> dict:
    """Build a dict of all config values for the given combination.

    Parameters
    ----------
    approach : str
        "pose", "video", or "fusion".
    variant : int
        100, 300, 1000, or 2000.
    tier : str
        "high", "mid", "low", or "cpu".
    hw : HardwareInfo
        Detected hardware info.

    Returns
    -------
    dict
        All config key-value pairs.
    """
    # --- Base values per approach ---
    if approach == "pose":
        cfg = {
            "approach": "pose_transformer",
            "wlasl_variant": variant,
            "num_keypoints": 543,
            "T": 64,
            "use_motion": True,
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dropout": 0.3,
            "num_workers": 4,
            "batch_size": 32,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "warmup_epochs": 10,
            "label_smoothing": 0.1,
            "grad_clip": 1.0,
            "fp16": True,
            "weighted_sampling": True,
            "early_stopping_patience": 20,
            "mixup_alpha": 0.2,
            "scheduler": "onecycle",
            "epochs": 100,
            "use_tta": False,
        }
    elif approach == "video":
        cfg = {
            "approach": "video",
            "backbone": "r2plus1d_18",
            "pretrained": True,
            "wlasl_variant": variant,
            "T": 32,
            "image_size": 224,
            "dropout": 0.4,
            "num_workers": 4,
            "batch_size": 8,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 10,
            "label_smoothing": 0.1,
            "grad_clip": 1.0,
            "fp16": True,
            "weighted_sampling": False,
            "early_stopping_patience": 20,
            "scheduler": "onecycle",
            "epochs": 100,
        }
    else:  # fusion
        cfg = {
            "approach": "fusion",
            "fusion": "concat",
            "fusion_dim": 256,
            "backbone": "r2plus1d_18",
            "pretrained": True,
            "num_keypoints": 543,
            "wlasl_variant": variant,
            "T": 64,
            "image_size": 224,
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dropout": 0.3,
            "num_workers": 4,
            "batch_size": 8,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 10,
            "label_smoothing": 0.1,
            "grad_clip": 1.0,
            "fp16": True,
            "weighted_sampling": False,
            "early_stopping_patience": 20,
            "scheduler": "onecycle",
            "epochs": 100,
        }

    # --- Variant-specific overrides (pose approach) ---
    if approach == "pose":
        if variant == 300:
            cfg.update({
                "num_layers": 6,
                "dropout": 0.25,
                "lr": 5e-4,
                "scheduler": "cosine",
                "early_stopping_patience": 25,
                "epochs": 150,
            })
        elif variant >= 1000:
            cfg.update({
                "d_model": 384,
                "num_layers": 6,
                "dropout": 0.2,
                "lr": 5e-4,
                "scheduler": "cosine",
                "warmup_epochs": 15,
                "early_stopping_patience": 30,
                "epochs": 200,
            })

    # --- Variant-specific overrides (video/fusion) ---
    if approach in ("video", "fusion") and variant >= 300:
        cfg["epochs"] = 150
        cfg["early_stopping_patience"] = 25
    if approach in ("video", "fusion") and variant >= 1000:
        cfg["epochs"] = 200
        cfg["early_stopping_patience"] = 30

    # --- Tier-specific overrides (hardware-dependent) ---
    tier_overrides = _get_tier_overrides(approach, tier, hw)
    cfg.update(tier_overrides)

    # Common settings across all configs
    cfg.update({
        "use_wandb": False,
        "use_tensorboard": True,
        "log_interval": 10,
        "confidence_threshold": 0.6,
        "smoothing_window": 5,
        "fps_display": True,
        "data_dir": "data",
        "output_dir": "outputs",
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
    })

    # buffer_size matches T
    cfg["buffer_size"] = cfg["T"]

    return cfg


def _get_tier_overrides(approach: str, tier: str, hw: HardwareInfo) -> dict:
    """Get hardware-dependent parameter overrides for a tier."""
    overrides: dict = {}

    if approach == "pose":
        batch_map = {"high": 64, "mid": 32, "low": 16, "cpu": 8}
        t_map = {"high": 64, "mid": 64, "low": 48, "cpu": 32}
        overrides["batch_size"] = batch_map[tier]
        overrides["T"] = t_map[tier]
    elif approach == "video":
        batch_map = {"high": 16, "mid": 8, "low": 4, "cpu": 4}
        t_map = {"high": 32, "mid": 32, "low": 16, "cpu": 16}
        img_map = {"high": 224, "mid": 224, "low": 112, "cpu": 112}
        overrides["batch_size"] = batch_map[tier]
        overrides["T"] = t_map[tier]
        overrides["image_size"] = img_map[tier]
    else:  # fusion
        batch_map = {"high": 16, "mid": 8, "low": 4, "cpu": 4}
        t_map = {"high": 64, "mid": 64, "low": 32, "cpu": 32}
        img_map = {"high": 224, "mid": 224, "low": 112, "cpu": 112}
        overrides["batch_size"] = batch_map[tier]
        overrides["T"] = t_map[tier]
        overrides["image_size"] = img_map[tier]

    # FP16 only on CUDA
    overrides["fp16"] = hw.device == "cuda"

    # num_workers based on device and cores
    if hw.device == "cuda":
        overrides["num_workers"] = min(8, hw.cpu_cores)
    else:
        overrides["num_workers"] = min(2, hw.cpu_cores)

    return overrides


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------


def render_yaml(approach: str, values: dict, hw: HardwareInfo, tier: str) -> str:
    """Render a formatted YAML config string with comments.

    Parameters
    ----------
    approach : str
        "pose", "video", or "fusion".
    values : dict
        Config values from ``build_config_values``.
    hw : HardwareInfo
        Hardware info for the header comment.
    tier : str
        Tier name for the header comment.

    Returns
    -------
    str
        YAML content ready to write to file.
    """
    # Header
    vram_str = f"{hw.vram_gb} GB VRAM, " if hw.vram_gb > 0 else ""
    cuda_str = f"CUDA {hw.cuda_version}" if hw.cuda_version else hw.device.upper()
    header = (
        f"## Auto-generated by scripts/auto_config.py\n"
        f"## Hardware: {hw.device_name} ({vram_str}{cuda_str})\n"
        f"## Tier: {tier} | Approach: {values['approach']} | "
        f"Variant: WLASL{values['wlasl_variant']}\n"
        f"## Re-run to regenerate, or edit manually.\n"
    )

    def _bool(v: bool) -> str:
        return "true" if v else "false"

    def _lr(v: float) -> str:
        return f"{v:.1e}"

    if approach == "pose":
        body = f"""\
approach: {values['approach']}
wlasl_variant: {values['wlasl_variant']}
# num_classes is auto-derived from wlasl_variant (100 -> 100, 300 -> 300, etc.)
num_keypoints: {values['num_keypoints']}

# Temporal
T: {values['T']}

# Features
use_motion: {_bool(values['use_motion'])}

# Model architecture
d_model: {values['d_model']}
nhead: {values['nhead']}
num_layers: {values['num_layers']}
dropout: {values['dropout']}

# Data loading
num_workers: {values['num_workers']}

# Training
epochs: {values['epochs']}
batch_size: {values['batch_size']}
lr: {_lr(values['lr'])}
weight_decay: {_lr(values['weight_decay'])}
warmup_epochs: {values['warmup_epochs']}
label_smoothing: {values['label_smoothing']}
grad_clip: {values['grad_clip']}
fp16: {_bool(values['fp16'])}
weighted_sampling: {_bool(values['weighted_sampling'])}
early_stopping_patience: {values['early_stopping_patience']}
mixup_alpha: {values['mixup_alpha']}

# Scheduler
scheduler: {values['scheduler']}

# Evaluation
use_tta: {_bool(values['use_tta'])}

# Logging
use_wandb: {_bool(values['use_wandb'])}
use_tensorboard: {_bool(values['use_tensorboard'])}
log_interval: {values['log_interval']}

# Inference
confidence_threshold: {values['confidence_threshold']}
smoothing_window: {values['smoothing_window']}
buffer_size: {values['buffer_size']}
fps_display: {_bool(values['fps_display'])}

# Paths
data_dir: {values['data_dir']}
output_dir: {values['output_dir']}
checkpoint_dir: {values['checkpoint_dir']}
log_dir: {values['log_dir']}
"""

    elif approach == "video":
        body = f"""\
approach: {values['approach']}
backbone: {values['backbone']}
pretrained: {_bool(values['pretrained'])}
wlasl_variant: {values['wlasl_variant']}
# num_classes is auto-derived from wlasl_variant (100 -> 100, 300 -> 300, etc.)

# Temporal & spatial
T: {values['T']}
image_size: {values['image_size']}

# Model
dropout: {values['dropout']}

# Data loading
num_workers: {values['num_workers']}

# Training
epochs: {values['epochs']}
batch_size: {values['batch_size']}
lr: {_lr(values['lr'])}
weight_decay: {_lr(values['weight_decay'])}
warmup_epochs: {values['warmup_epochs']}
label_smoothing: {values['label_smoothing']}
grad_clip: {values['grad_clip']}
fp16: {_bool(values['fp16'])}
weighted_sampling: {_bool(values['weighted_sampling'])}
early_stopping_patience: {values['early_stopping_patience']}

# Scheduler
scheduler: {values['scheduler']}

# Logging
use_wandb: {_bool(values['use_wandb'])}
use_tensorboard: {_bool(values['use_tensorboard'])}
log_interval: {values['log_interval']}

# Inference
confidence_threshold: {values['confidence_threshold']}
smoothing_window: {values['smoothing_window']}
buffer_size: {values['buffer_size']}
fps_display: {_bool(values['fps_display'])}

# Paths
data_dir: {values['data_dir']}
output_dir: {values['output_dir']}
checkpoint_dir: {values['checkpoint_dir']}
log_dir: {values['log_dir']}
"""

    else:  # fusion
        body = f"""\
approach: {values['approach']}
fusion: {values['fusion']}
fusion_dim: {values['fusion_dim']}

# Sub-model configs
backbone: {values['backbone']}
pretrained: {_bool(values['pretrained'])}
num_keypoints: {values['num_keypoints']}

# Dataset
wlasl_variant: {values['wlasl_variant']}
# num_classes is auto-derived from wlasl_variant (100 -> 100, 300 -> 300, etc.)
T: {values['T']}
image_size: {values['image_size']}

# Pose Transformer settings
d_model: {values['d_model']}
nhead: {values['nhead']}
num_layers: {values['num_layers']}
dropout: {values['dropout']}

# Data loading
num_workers: {values['num_workers']}

# Training
epochs: {values['epochs']}
batch_size: {values['batch_size']}
lr: {_lr(values['lr'])}
weight_decay: {_lr(values['weight_decay'])}
warmup_epochs: {values['warmup_epochs']}
label_smoothing: {values['label_smoothing']}
grad_clip: {values['grad_clip']}
fp16: {_bool(values['fp16'])}
weighted_sampling: {_bool(values['weighted_sampling'])}
early_stopping_patience: {values['early_stopping_patience']}

# Scheduler
scheduler: {values['scheduler']}

# Logging
use_wandb: {_bool(values['use_wandb'])}
use_tensorboard: {_bool(values['use_tensorboard'])}
log_interval: {values['log_interval']}

# Inference
confidence_threshold: {values['confidence_threshold']}
smoothing_window: {values['smoothing_window']}
buffer_size: {values['buffer_size']}
fps_display: {_bool(values['fps_display'])}

# Paths
data_dir: {values['data_dir']}
output_dir: {values['output_dir']}
checkpoint_dir: {values['checkpoint_dir']}
log_dir: {values['log_dir']}
"""

    return header + "\n" + body


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary(
    hw: HardwareInfo,
    tier: str,
    approach: str,
    variant: int,
    values: dict,
    output_path: Path,
) -> None:
    """Print a human-readable hardware detection and config summary."""
    print()
    print("=" * 60)
    print("  WLASL Auto-Config")
    print("=" * 60)
    print()
    print("  Hardware")
    print("  " + "-" * 40)
    print(f"  Device          : {hw.device_name}")
    if hw.vram_gb > 0:
        print(f"  VRAM            : {hw.vram_gb} GB")
    if hw.cuda_version:
        print(f"  CUDA            : {hw.cuda_version}")
    if hw.gpu_count > 1:
        print(f"  GPU count       : {hw.gpu_count}")
    print(f"  CPU cores       : {hw.cpu_cores}")
    print(f"  PyTorch         : {hw.torch_version}")
    print(f"  Platform        : {hw.platform_name}")
    print()
    print("  Configuration")
    print("  " + "-" * 40)
    print(f"  Tier            : {tier}")
    print(f"  Approach        : {values['approach']}")
    print(f"  WLASL variant   : {variant}")
    print(f"  Output file     : {output_path}")
    print()
    print("  Key Parameters (hardware-optimized)")
    print("  " + "-" * 40)
    print(f"  batch_size      : {values['batch_size']}")
    print(f"  T               : {values['T']}")
    print(f"  fp16            : {values['fp16']}")
    print(f"  num_workers     : {values['num_workers']}")
    if "image_size" in values and approach != "pose":
        print(f"  image_size      : {values['image_size']}")
    print(f"  lr              : {values['lr']}")
    print(f"  epochs          : {values['epochs']}")
    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect hardware and generate an optimized YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/auto_config.py --approach pose
  python scripts/auto_config.py --approach video --variant 100
  python scripts/auto_config.py --approach fusion --variant 300 --dry-run
  python scripts/auto_config.py --approach pose --device cpu
  python scripts/auto_config.py --approach pose --backup
""",
    )
    parser.add_argument(
        "--approach",
        type=str,
        required=True,
        choices=list(VALID_APPROACHES),
        help="Model approach: pose, video, or fusion",
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=100,
        choices=list(VALID_VARIANTS),
        help="WLASL variant (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Override detected device (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the generated config without writing to disk",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Back up existing config to {filename}.bak before overwriting",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path (default: configs/{approach}.yaml)",
    )
    args = parser.parse_args()

    # Detect hardware
    hw = detect_hardware(args.device)
    tier = determine_tier(hw)

    # Build config
    values = build_config_values(args.approach, args.variant, tier, hw)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = CONFIGS_DIR / APPROACH_TO_FILE[args.approach]

    # Render YAML
    yaml_content = render_yaml(args.approach, values, hw, tier)

    # Print summary
    print_summary(hw, tier, args.approach, args.variant, values, output_path)

    if args.dry_run:
        print("  [DRY RUN] Generated config:\n")
        print(yaml_content)
        return

    # Backup if requested
    if args.backup and output_path.exists():
        bak_path = output_path.with_suffix(".yaml.bak")
        shutil.copy2(output_path, bak_path)
        print(f"  Backed up {output_path} -> {bak_path}")

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_content, encoding="utf-8")
    print(f"  Wrote {output_path}")

    # Check if data exists
    splits_dir = PROJECT_ROOT / "data" / "splits" / f"WLASL{args.variant}"
    if not splits_dir.exists():
        print()
        print(f"  Note: {splits_dir.relative_to(PROJECT_ROOT)} not found.")
        print(f"  Run preprocessing first:")
        print(f"    python -m src.data.preprocess --data-dir data --subset WLASL{args.variant}")

    print()


if __name__ == "__main__":
    main()
