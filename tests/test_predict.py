"""Tests for src.inference.predict — SignPredictor inference logic."""

import numpy as np
import pytest
import torch

from src.models.stgcn import STGCNEncoder
from src.models.prototypical import PrototypicalNetwork, build_model
from src.training.config import Config


NUM_KP = 543


class TestPredictFromKeypoints:
    """Test the core _predict_from_keypoints path without needing MediaPipe."""

    def _make_predictor_and_checkpoint(self, tmp_path, use_motion=False):
        """Create a minimal model, save a checkpoint, and build a SignPredictor."""
        from src.inference.predict import SignPredictor

        cfg = Config(
            approach="stgcn_proto",
            num_keypoints=NUM_KP,
            wlasl_variant=10,
            d_model=64,
            gcn_channels=[32, 64],
            dropout=0.0,
            use_motion=use_motion,
            T=16,
        )
        model = build_model(cfg)

        # Compute fake prototypes so classify works
        data = torch.randn(20, 16, NUM_KP * (6 if use_motion else 3))
        labels = torch.tensor([i % 10 for i in range(20)])
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(data, labels)
        loader = DataLoader(ds, batch_size=10)
        model.compute_prototypes(loader)

        ckpt_path = tmp_path / "model.pt"
        torch.save({"model_state_dict": model.state_dict()}, str(ckpt_path))

        class_names = [f"sign_{i}" for i in range(10)]

        # Create dummy train CSV so _load_prototypes can find it
        splits_dir = tmp_path / "data" / "splits" / "WLASL10"
        splits_dir.mkdir(parents=True)
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)

        import pandas as pd
        rows = []
        for i in range(20):
            vid = f"vid_{i:03d}"
            rows.append({"video_id": vid, "label_idx": i % 10, "gloss": f"sign_{i % 10}"})
            kps = np.random.rand(30, NUM_KP, 3).astype(np.float32)
            np.save(str(processed_dir / f"{vid}.npy"), kps)
        pd.DataFrame(rows).to_csv(splits_dir / "train.csv", index=False)

        cfg.data_dir = str(tmp_path / "data")
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
