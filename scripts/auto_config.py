"""
Auto-detect hardware and generate an optimized YAML config.

Probes your GPU (CUDA), Apple Silicon (MPS), or CPU, determines a
performance tier, and writes a ready-to-train config for the ST-GCN +
Prototypical Network approach.

Usage:
    python scripts/auto_config.py
    python scripts/auto_config.py --variant 100
    python scripts/auto_config.py --dry-run
    python scripts/auto_config.py --device cpu
    python scripts/auto_config.py --backup
"""

import argparse
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

VALID_VARIANTS = (100, 300, 1000, 2000)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


@dataclass
class HardwareInfo:
    """Detected hardware information."""

    device: str
    device_name: str
    vram_gb: float
    cuda_version: str
    cpu_cores: int
    platform_name: str
    torch_version: str
    gpu_count: int


def detect_hardware(device_override: str | None = None) -> HardwareInfo:
    """Auto-detect the available hardware."""
    cpu_cores = os.cpu_count() or 1
    plat = platform.platform()

    try:
        import torch
    except ImportError:
        return HardwareInfo(
            device="cpu",
            device_name=platform.processor() or "Unknown CPU",
            vram_gb=0.0, cuda_version="", cpu_cores=cpu_cores,
            platform_name=plat, torch_version="(not installed)", gpu_count=0,
        )

    torch_ver = torch.__version__

    if device_override:
        device = device_override
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        vram_gb = props.total_memory / (1024**3)
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
        device=device, device_name=device_name,
        vram_gb=round(vram_gb, 1), cuda_version=cuda_version,
        cpu_cores=cpu_cores, platform_name=plat,
        torch_version=torch_ver, gpu_count=gpu_count,
    )


def determine_tier(hw: HardwareInfo) -> str:
    """Classify hardware into a performance tier."""
    if hw.device == "cuda":
        if hw.vram_gb >= 16:
            return "high"
        elif hw.vram_gb >= 8:
            return "mid"
        else:
            return "low"
    elif hw.device == "mps":
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Config value generation
# ---------------------------------------------------------------------------


def build_config_values(variant: int, tier: str, hw: HardwareInfo) -> dict:
    """Build a dict of all config values for the given combination."""
    cfg = {
        "approach": "stgcn_proto",
        "wlasl_variant": variant,
        "num_keypoints": 543,
        "T": 64,
        "use_motion": True,
        "d_model": 128,
        "gcn_channels": [64, 128, 128],
        "num_layers": 3,
        "dropout": 0.1,
        "n_way": 20,
        "k_shot": 3,
        "q_query": 2,
        "num_episodes": 500,
        "num_workers": 4,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 10,
        "grad_clip": 1.0,
        "fp16": False,
        "weighted_sampling": False,
        "early_stopping_patience": 30,
        "scheduler": "cosine",
        "epochs": 200,
        "use_tta": False,
    }

    # Variant-specific architecture scaling
    if variant == 100:
        cfg.update({
            "gcn_channels": [64, 128, 128],
            "d_model": 128,
            "dropout": 0.1,
        })
    elif variant == 300:
        cfg.update({
            "gcn_channels": [64, 128, 256],
            "d_model": 192,
            "dropout": 0.15,
            "epochs": 250,
        })
    elif variant == 1000:
        cfg.update({
            "gcn_channels": [64, 128, 256],
            "d_model": 256,
            "dropout": 0.2,
            "epochs": 300,
        })
    elif variant >= 2000:
        cfg.update({
            "gcn_channels": [64, 128, 256, 256],
            "d_model": 384,
            "dropout": 0.2,
            "epochs": 350,
        })

    # Tier-specific overrides
    batch_map = {"high": 64, "mid": 32, "low": 16, "mps": 16, "cpu": 8}
    cfg["batch_size"] = batch_map[tier]
    cfg["fp16"] = hw.device == "cuda"

    if hw.device == "cuda":
        cfg["num_workers"] = min(8, hw.cpu_cores)
    elif hw.device == "mps":
        # MPS deadlocks with multiprocessing DataLoader workers
        cfg["num_workers"] = 0
        # Lower n_way and num_episodes to avoid Metal command queue stalls
        cfg["n_way"] = 10
        cfg["num_episodes"] = 200
    else:
        cfg["num_workers"] = min(2, hw.cpu_cores)

    # Common settings
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
        "buffer_size": cfg["T"],
    })

    return cfg


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------


def render_yaml(values: dict, hw: HardwareInfo, tier: str) -> str:
    """Render a formatted YAML config string with comments."""
    vram_str = f"{hw.vram_gb} GB VRAM, " if hw.vram_gb > 0 else ""
    cuda_str = f"CUDA {hw.cuda_version}" if hw.cuda_version else hw.device.upper()
    header = (
        f"## Auto-generated by scripts/auto_config.py\n"
        f"## Hardware: {hw.device_name} ({vram_str}{cuda_str})\n"
        f"## Tier: {tier} | Variant: WLASL{values['wlasl_variant']}\n"
        f"## Re-run to regenerate, or edit manually.\n"
    )

    def _bool(v: bool) -> str:
        return "true" if v else "false"

    def _lr(v: float) -> str:
        return f"{v:.1e}"

    channels_str = "[" + ", ".join(str(c) for c in values["gcn_channels"]) + "]"

    body = f"""\
approach: {values['approach']}
wlasl_variant: {values['wlasl_variant']}
num_keypoints: {values['num_keypoints']}

# Temporal
T: {values['T']}

# Features
use_motion: {_bool(values['use_motion'])}

# Model (ST-GCN)
d_model: {values['d_model']}
gcn_channels: {channels_str}
num_layers: {values['num_layers']}
dropout: {values['dropout']}

# Prototypical training
n_way: {values['n_way']}
k_shot: {values['k_shot']}
q_query: {values['q_query']}
num_episodes: {values['num_episodes']}

# Data loading
num_workers: {values['num_workers']}

# Training
epochs: {values['epochs']}
batch_size: {values['batch_size']}
lr: {_lr(values['lr'])}
weight_decay: {_lr(values['weight_decay'])}
warmup_epochs: {values['warmup_epochs']}
grad_clip: {values['grad_clip']}
fp16: {_bool(values['fp16'])}
early_stopping_patience: {values['early_stopping_patience']}

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

    return header + "\n" + body


def print_summary(
    hw: HardwareInfo, tier: str, variant: int, values: dict, output_path: Path,
) -> None:
    """Print a human-readable hardware detection and config summary."""
    print()
    print("=" * 60)
    print("  WLASL Auto-Config (ST-GCN + Prototypical)")
    print("=" * 60)
    print()
    print("  Hardware")
    print("  " + "-" * 40)
    print(f"  Device          : {hw.device_name}")
    if hw.vram_gb > 0:
        print(f"  VRAM            : {hw.vram_gb} GB")
    if hw.cuda_version:
        print(f"  CUDA            : {hw.cuda_version}")
    print(f"  CPU cores       : {hw.cpu_cores}")
    print(f"  PyTorch         : {hw.torch_version}")
    print()
    print("  Configuration")
    print("  " + "-" * 40)
    print(f"  Tier            : {tier}")
    print(f"  WLASL variant   : {variant}")
    print(f"  Output file     : {output_path}")
    print()
    print("  Key Parameters")
    print("  " + "-" * 40)
    print(f"  batch_size      : {values['batch_size']}")
    print(f"  n_way / k_shot  : {values['n_way']}-way {values['k_shot']}-shot")
    print(f"  num_episodes    : {values['num_episodes']}")
    print(f"  d_model         : {values['d_model']}")
    print(f"  fp16            : {values['fp16']}")
    print(f"  epochs          : {values['epochs']}")
    print()
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect hardware and generate an optimized YAML config",
    )
    parser.add_argument(
        "--variant", type=int, default=100, choices=list(VALID_VARIANTS),
        help="WLASL variant (default: 100)",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
        help="Override detected device (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview the generated config without writing to disk",
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="Back up existing config before overwriting",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Custom output path (default: configs/stgcn_proto.yaml)",
    )
    args = parser.parse_args()

    hw = detect_hardware(args.device)
    tier = determine_tier(hw)

    values = build_config_values(args.variant, tier, hw)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = CONFIGS_DIR / "stgcn_proto.yaml"

    yaml_content = render_yaml(values, hw, tier)

    print_summary(hw, tier, args.variant, values, output_path)

    if args.dry_run:
        print("  [DRY RUN] Generated config:\n")
        print(yaml_content)
        return

    if args.backup and output_path.exists():
        bak_path = output_path.with_suffix(".yaml.bak")
        shutil.copy2(output_path, bak_path)
        print(f"  Backed up {output_path} -> {bak_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_content, encoding="utf-8")
    print(f"  Wrote {output_path}")

    splits_dir = PROJECT_ROOT / "data" / "splits" / f"WLASL{args.variant}"
    if not splits_dir.exists():
        print()
        print(f"  Note: {splits_dir.relative_to(PROJECT_ROOT)} not found.")
        print(f"  Run preprocessing first:")
        print(f"    python -m src.data.preprocess --data-dir data --subset WLASL{args.variant}")

    print()


if __name__ == "__main__":
    main()
