"""Tests for src.data.dataset — WLASLKeypointDataset, get_dataloader."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import (
    WLASLKeypointDataset,
    get_dataloader,
)

NUM_KP = 543


# ---------------------------------------------------------------------------
# WLASLKeypointDataset._pad_or_crop
# ---------------------------------------------------------------------------


class TestPadOrCrop:
    def _make_ds(self, tmp_path):
        """Create a minimal dataset for testing."""
        kp_dir = tmp_path / "kps"
        kp_dir.mkdir()
        df = pd.DataFrame({
            "video_id": ["v0"],
            "label_idx": [0],
            "gloss": ["test"],
        })
        csv = tmp_path / "split.csv"
        df.to_csv(csv, index=False)
        kps = np.random.rand(30, NUM_KP, 3).astype(np.float32)
        np.save(str(kp_dir / "v0.npy"), kps)
        return WLASLKeypointDataset(csv, kp_dir, T=16)

    def test_exact_length(self, tmp_path):
        ds = self._make_ds(tmp_path)
        kps = np.random.rand(16, NUM_KP, 3).astype(np.float32)
        result = ds._pad_or_crop(kps)
        assert result.shape[0] == 16

    def test_longer_crops(self, tmp_path):
        ds = self._make_ds(tmp_path)
        kps = np.random.rand(100, NUM_KP, 3).astype(np.float32)
        result = ds._pad_or_crop(kps)
        assert result.shape == (16, NUM_KP, 3)

    def test_shorter_pads(self, tmp_path):
        ds = self._make_ds(tmp_path)
        kps = np.random.rand(10, NUM_KP, 3).astype(np.float32)
        result = ds._pad_or_crop(kps)
        assert result.shape == (16, NUM_KP, 3)
        # Padding should duplicate last frame
        np.testing.assert_array_equal(result[10], result[9])

    def test_empty_input(self, tmp_path):
        ds = self._make_ds(tmp_path)
        kps = np.zeros((0, NUM_KP, 3), dtype=np.float32)
        result = ds._pad_or_crop(kps)
        assert result.shape == (16, NUM_KP, 3)
        assert np.all(result == 0)


# ---------------------------------------------------------------------------
# WLASLKeypointDataset
# ---------------------------------------------------------------------------


class TestWLASLKeypointDataset:
    def test_len(self, tmp_dataset):
        csv_path, kp_dir = tmp_dataset
        ds = WLASLKeypointDataset(csv_path, kp_dir, T=16)
        assert len(ds) == 4

    def test_getitem_shape_no_motion(self, tmp_dataset):
        csv_path, kp_dir = tmp_dataset
        ds = WLASLKeypointDataset(csv_path, kp_dir, T=16, use_motion=False)
        tensor, label = ds[0]
        assert tensor.shape == (16, NUM_KP * 3)
        assert isinstance(label, int)

    def test_getitem_shape_with_motion(self, tmp_dataset):
        csv_path, kp_dir = tmp_dataset
        ds = WLASLKeypointDataset(csv_path, kp_dir, T=16, use_motion=True)
        tensor, label = ds[0]
        assert tensor.shape == (16, NUM_KP * 6)

    def test_motion_first_frame_velocity_zero(self, tmp_dataset):
        csv_path, kp_dir = tmp_dataset
        ds = WLASLKeypointDataset(csv_path, kp_dir, T=16, use_motion=True)
        tensor, _ = ds[0]
        # Velocity part of first frame should be zero
        # Features are (x,y,z,dx,dy,dz) per keypoint, so velocity = indices 3,4,5 per keypoint
        frame0 = tensor[0].numpy().reshape(NUM_KP, 6)
        velocity_frame0 = frame0[:, 3:]
        np.testing.assert_allclose(velocity_frame0, 0.0, atol=1e-7)

    def test_labels_correct(self, tmp_dataset):
        csv_path, kp_dir = tmp_dataset
        ds = WLASLKeypointDataset(csv_path, kp_dir, T=16)
        labels = [ds[i][1] for i in range(len(ds))]
        assert set(labels) == {0, 1}

    def test_missing_npy_filtered(self, tmp_path):
        """Samples without .npy files should be filtered out."""
        kp_dir = tmp_path / "kps"
        kp_dir.mkdir()
        # Create CSV with 3 entries, but only 1 npy file
        df = pd.DataFrame({
            "video_id": ["exists", "missing1", "missing2"],
            "label_idx": [0, 1, 2],
            "gloss": ["a", "b", "c"],
        })
        csv = tmp_path / "split.csv"
        df.to_csv(csv, index=False)
        kps = np.random.rand(20, NUM_KP, 3).astype(np.float32)
        np.save(str(kp_dir / "exists.npy"), kps)

        ds = WLASLKeypointDataset(csv, kp_dir, T=16)
        assert len(ds) == 1

    def test_tensor_dtype_float(self, tmp_dataset):
        csv_path, kp_dir = tmp_dataset
        ds = WLASLKeypointDataset(csv_path, kp_dir, T=16)
        tensor, _ = ds[0]
        assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# get_dataloader
# ---------------------------------------------------------------------------


class TestGetDataloader:
    def test_basic(self, tmp_dataset):
        csv_path, kp_dir = tmp_dataset
        ds = WLASLKeypointDataset(csv_path, kp_dir, T=16)
        loader = get_dataloader(ds, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        assert batch[0].shape[0] <= 2  # batch dim

    def test_weighted_sampling(self, tmp_dataset):
        csv_path, kp_dir = tmp_dataset
        ds = WLASLKeypointDataset(csv_path, kp_dir, T=16)
        loader = get_dataloader(ds, batch_size=2, num_workers=0, weighted_sampling=True)
        batch = next(iter(loader))
        assert batch[0].shape[0] <= 2
