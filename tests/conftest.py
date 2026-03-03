"""Shared pytest fixtures for the WLASL test suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Random keypoint helpers
# ---------------------------------------------------------------------------

NUM_KEYPOINTS = 543


def _make_keypoints(T: int = 30, K: int = NUM_KEYPOINTS, C: int = 3) -> np.ndarray:
    """Generate random keypoints shaped (T, K, C) with realistic shoulder positions."""
    kps = np.random.rand(T, K, C).astype(np.float32)
    # Set shoulder landmarks (11, 12) to consistent positions so normalization works
    kps[:, 11, :] = [0.4, 0.5, 0.0]
    kps[:, 12, :] = [0.6, 0.5, 0.0]
    return kps


@pytest.fixture
def random_keypoints_3d():
    """Return random (30, 543, 3) keypoint array."""
    return _make_keypoints(T=30)


@pytest.fixture
def random_keypoints_flat():
    """Return random (30, 1629) flat keypoint array."""
    return _make_keypoints(T=30).reshape(30, -1)


# ---------------------------------------------------------------------------
# Temporary dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a tiny on-disk dataset: CSV + .npy files for 4 samples, 2 classes."""
    kp_dir = tmp_path / "processed"
    kp_dir.mkdir()
    split_dir = tmp_path / "splits"
    split_dir.mkdir()

    rows = []
    for i in range(4):
        vid = f"vid_{i:03d}"
        label = i % 2
        gloss = f"sign_{label}"
        rows.append({"video_id": vid, "label_idx": label, "gloss": gloss})
        kps = _make_keypoints(T=np.random.randint(20, 50))
        np.save(str(kp_dir / f"{vid}.npy"), kps)

    df = pd.DataFrame(rows)
    csv_path = split_dir / "train.csv"
    df.to_csv(csv_path, index=False)

    return csv_path, kp_dir


@pytest.fixture
def tmp_config_yaml(tmp_path):
    """Write a minimal YAML config file and return its path."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "approach: pose_transformer\n"
        "wlasl_variant: 100\n"
        "T: 16\n"
        "d_model: 64\n"
        "nhead: 4\n"
        "num_layers: 2\n"
        "dropout: 0.1\n"
        "use_motion: false\n"
        "mixup_alpha: 0.0\n"
        "use_tta: false\n"
    )
    return cfg_path
