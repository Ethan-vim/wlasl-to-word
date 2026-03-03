"""Tests for src.data.preprocess — normalization, parsing, splits."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    NUM_KEYPOINTS,
    WLASL_VARIANT_SIZES,
    create_splits,
    normalize_keypoints,
    parse_wlasl_annotations,
)


# ---------------------------------------------------------------------------
# normalize_keypoints
# ---------------------------------------------------------------------------


class TestNormalizeKeypoints:
    def _make_kps(self, T=20):
        kps = np.random.rand(T, NUM_KEYPOINTS, 3).astype(np.float32)
        kps[:, 11, :] = [0.4, 0.5, 0.0]  # left shoulder
        kps[:, 12, :] = [0.6, 0.5, 0.0]  # right shoulder
        return kps

    def test_shape_preserved(self):
        kps = self._make_kps()
        result = normalize_keypoints(kps)
        assert result.shape == kps.shape

    def test_shoulder_midpoint_centered(self):
        """After normalization, shoulder midpoint should be at the origin."""
        kps = self._make_kps(T=10)
        result = normalize_keypoints(kps)
        shoulder_mid = (result[:, 11, :] + result[:, 12, :]) / 2.0
        np.testing.assert_allclose(shoulder_mid, 0.0, atol=1e-5)

    def test_shoulder_width_normalized(self):
        """After normalization, shoulder width should be ~1."""
        kps = self._make_kps(T=10)
        result = normalize_keypoints(kps)
        widths = np.linalg.norm(result[:, 11, :] - result[:, 12, :], axis=-1)
        np.testing.assert_allclose(widths, 1.0, atol=1e-5)

    def test_zero_frame_interpolation(self):
        """Zero-padded frames should be replaced by nearest valid frame."""
        kps = self._make_kps(T=10)
        kps[3] = 0.0  # simulate detection failure
        kps[4] = 0.0
        result = normalize_keypoints(kps)
        # After interpolation, frame 3 and 4 should NOT be all-zero
        assert np.linalg.norm(result[3]) > 1e-6
        assert np.linalg.norm(result[4]) > 1e-6

    def test_all_zero_frames_no_crash(self):
        """All-zero input should not crash."""
        kps = np.zeros((5, NUM_KEYPOINTS, 3), dtype=np.float32)
        result = normalize_keypoints(kps)
        assert result.shape == kps.shape

    def test_does_not_modify_input(self):
        kps = self._make_kps(T=5)
        original = kps.copy()
        normalize_keypoints(kps)
        np.testing.assert_array_equal(kps, original)


# ---------------------------------------------------------------------------
# parse_wlasl_annotations
# ---------------------------------------------------------------------------


class TestParseAnnotations:
    def _make_json(self, tmp_path, num_glosses=5, instances_per=3):
        data = []
        for i in range(num_glosses):
            instances = []
            for j in range(instances_per):
                instances.append({
                    "video_id": f"vid_{i}_{j}",
                    "split": ["train", "val", "test"][j % 3],
                    "signer_id": j,
                    "fps": 25,
                    "url": f"http://example.com/{i}_{j}",
                })
            data.append({"gloss": f"sign_{i}", "instances": instances})

        json_path = tmp_path / "annotations.json"
        with open(json_path, "w") as f:
            json.dump(data, f)
        return json_path

    def test_basic_parsing(self, tmp_path):
        json_path = self._make_json(tmp_path, num_glosses=200, instances_per=2)
        # WLASL100 takes first 100 glosses
        df = parse_wlasl_annotations(json_path, subset="WLASL100")
        assert len(df) == 200  # 100 glosses * 2 instances
        assert df["label_idx"].max() == 99
        assert set(df.columns) >= {"video_id", "gloss", "label_idx", "split"}

    def test_invalid_subset_raises(self, tmp_path):
        json_path = self._make_json(tmp_path)
        with pytest.raises(ValueError, match="Unknown subset"):
            parse_wlasl_annotations(json_path, subset="WLASL50")


# ---------------------------------------------------------------------------
# create_splits
# ---------------------------------------------------------------------------


class TestCreateSplits:
    def test_creates_csv_files(self, tmp_path):
        df = pd.DataFrame({
            "video_id": ["a", "b", "c"],
            "gloss": ["x", "x", "y"],
            "label_idx": [0, 0, 1],
            "split": ["train", "val", "test"],
        })
        paths = create_splits(df, tmp_path / "splits")
        assert (tmp_path / "splits" / "train.csv").exists()
        assert (tmp_path / "splits" / "val.csv").exists()
        assert (tmp_path / "splits" / "test.csv").exists()

    def test_split_content(self, tmp_path):
        df = pd.DataFrame({
            "video_id": ["a", "b", "c", "d"],
            "gloss": ["x", "x", "y", "y"],
            "label_idx": [0, 0, 1, 1],
            "split": ["train", "train", "val", "test"],
        })
        paths = create_splits(df, tmp_path / "out")
        train_df = pd.read_csv(paths["train"])
        assert len(train_df) == 2
        val_df = pd.read_csv(paths["val"])
        assert len(val_df) == 1
