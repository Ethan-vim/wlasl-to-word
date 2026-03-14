"""
Configuration dataclass and YAML serialization for training.

All hyperparameters, paths, and settings are centralized in the ``Config``
dataclass.  Configurations can be loaded from and saved to YAML files.
"""

import logging
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Central configuration for the WLASL recognition system.

    All fields have sensible defaults so that a minimal YAML file only
    needs to override the values that differ from the defaults.
    """

    # --- Paths ---
    data_dir: str = "data"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # --- Dataset ---
    wlasl_variant: int = 100
    T: int = 64
    num_workers: int = 4

    # --- Model (ST-GCN) ---
    approach: str = "stgcn_proto"
    num_keypoints: int = 543
    num_classes: int = 100
    d_model: int = 128  # embedding dimension (alias for embedding_dim)
    embedding_dim: int = 128
    gcn_channels: list[int] = field(default_factory=lambda: [64, 128, 128])
    num_layers: int = 3  # number of ST-GCN blocks per branch
    dropout: float = 0.1

    # --- Features ---
    use_motion: bool = True

    # --- Prototypical training ---
    n_way: int = 20
    k_shot: int = 3
    q_query: int = 2
    num_episodes: int = 500

    # --- Training ---
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    fp16: bool = False
    weighted_sampling: bool = False
    early_stopping_patience: int = 30

    # --- Scheduler ---
    scheduler: str = "cosine"

    # --- Logging ---
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: str = "wlasl-recognition"
    wandb_run_name: Optional[str] = None
    log_interval: int = 10

    # --- Evaluation ---
    use_tta: bool = False

    # --- Inference ---
    confidence_threshold: float = 0.6
    smoothing_window: int = 5
    buffer_size: int = 64
    fps_display: bool = True

    # --- Resume ---
    resume_checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        """Derive num_classes from wlasl_variant and sync embedding_dim."""
        variant_to_classes = {100: 100, 300: 300, 1000: 1000, 2000: 2000}
        if self.wlasl_variant in variant_to_classes:
            self.num_classes = variant_to_classes[self.wlasl_variant]

        # Keep embedding_dim and d_model in sync
        self.embedding_dim = self.d_model

        # Scale ST-GCN channels by variant
        variant_to_arch = {
            100:  {"gcn_channels": [64, 128, 128], "d_model": 128, "dropout": 0.1},
            300:  {"gcn_channels": [64, 128, 256], "d_model": 192, "dropout": 0.15},
            1000: {"gcn_channels": [64, 128, 256], "d_model": 256, "dropout": 0.2},
            2000: {"gcn_channels": [64, 128, 256, 256], "d_model": 384, "dropout": 0.2},
        }
        if self.wlasl_variant in variant_to_arch:
            arch = variant_to_arch[self.wlasl_variant]
            self.gcn_channels = arch["gcn_channels"]
            self.d_model = arch["d_model"]
            self.embedding_dim = arch["d_model"]
            self.dropout = arch["dropout"]


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file.

    Keys in the YAML file that match Config field names will override
    the defaults.  Unknown keys are logged as warnings and ignored.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    Config
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    valid_fields = {f.name for f in fields(Config)}
    kwargs = {}
    for key, value in raw.items():
        if key in valid_fields:
            kwargs[key] = value
        else:
            logger.warning("Unknown config key '%s' in %s (ignored)", key, path)

    cfg = Config(**kwargs)
    logger.info("Loaded config from %s", path)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Save a Config to a YAML file.

    Parameters
    ----------
    cfg : Config
        The configuration to save.
    path : str or Path
        Destination YAML file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(cfg)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    logger.info("Saved config to %s", path)
