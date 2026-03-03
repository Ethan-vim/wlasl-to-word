"""Tests for src.inference.predict — SignPredictor inference logic."""

import numpy as np
import pytest
import torch

from src.models.pose_transformer import PoseTransformer
from src.training.config import Config


NUM_KP = 543


class TestPredictFromKeypoints:
    """Test the core _predict_from_keypoints path without needing MediaPipe."""

    def _make_predictor_and_checkpoint(self, tmp_path, use_motion=False):
        """Create a minimal model, save a checkpoint, and build a SignPredictor."""
        from src.inference.predict import SignPredictor

        cfg = Config(
            approach="pose_transformer",
            num_keypoints=NUM_KP,
            num_classes=10,
            wlasl_variant=10,
            d_model=64,
            nhead=4,
            num_layers=1,
            T=16,
            dropout=0.0,
            use_motion=use_motion,
        )
        model = PoseTransformer(
            num_keypoints=NUM_KP, num_classes=10, d_model=64,
            nhead=4, num_layers=1, T=16, use_motion=use_motion,
        )
        ckpt_path = tmp_path / "model.pt"
        torch.save({"model_state_dict": model.state_dict()}, str(ckpt_path))

        class_names = [f"sign_{i}" for i in range(10)]
        predictor = SignPredictor(
            checkpoint_path=ckpt_path, cfg=cfg, device="cpu",
            class_names=class_names,
        )
        return predictor

    def test_predict_keypoints_file(self, tmp_path):
        predictor = self._make_predictor_and_checkpoint(tmp_path)
        # Create a dummy .npy file
        kps = np.random.rand(30, NUM_KP, 3).astype(np.float32)
        npy_path = tmp_path / "sample.npy"
        np.save(str(npy_path), kps)

        result = predictor.predict_keypoints(npy_path)
        assert "gloss" in result
        assert "confidence" in result
        assert "top5" in result
        assert "label_idx" in result
        assert len(result["top5"]) == 5
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_keypoints_with_motion(self, tmp_path):
        predictor = self._make_predictor_and_checkpoint(tmp_path, use_motion=True)
        kps = np.random.rand(30, NUM_KP, 3).astype(np.float32)
        npy_path = tmp_path / "sample.npy"
        np.save(str(npy_path), kps)

        result = predictor.predict_keypoints(npy_path)
        assert "gloss" in result
        assert result["gloss"].startswith("sign_")

    def test_short_sequence(self, tmp_path):
        """A very short sequence should still produce valid output (padded)."""
        predictor = self._make_predictor_and_checkpoint(tmp_path)
        kps = np.random.rand(3, NUM_KP, 3).astype(np.float32)
        npy_path = tmp_path / "short.npy"
        np.save(str(npy_path), kps)

        result = predictor.predict_keypoints(npy_path)
        assert "gloss" in result

    def test_confidence_sums_to_one(self, tmp_path):
        """Top-5 probabilities should be <= 1 each."""
        predictor = self._make_predictor_and_checkpoint(tmp_path)
        kps = np.random.rand(30, NUM_KP, 3).astype(np.float32)
        npy_path = tmp_path / "sample.npy"
        np.save(str(npy_path), kps)

        result = predictor.predict_keypoints(npy_path)
        for _, prob in result["top5"]:
            assert 0.0 <= prob <= 1.0
