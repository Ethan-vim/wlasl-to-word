"""Tests for src.training.evaluate — metrics, TTA, hard negatives, latency."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.evaluate import (
    _flip_keypoints_tensor,
    compute_metrics,
    evaluate_latency,
    find_hard_negatives,
    plot_confusion_matrix,
)

NUM_KP = 543


# ---------------------------------------------------------------------------
# _flip_keypoints_tensor
# ---------------------------------------------------------------------------


class TestFlipKeypointsTensor:
    def test_shape_preserved_3c(self):
        x = torch.randn(2, 16, NUM_KP * 3)
        result = _flip_keypoints_tensor(x, num_keypoints=NUM_KP)
        assert result.shape == x.shape

    def test_shape_preserved_6c(self):
        x = torch.randn(2, 16, NUM_KP * 6)
        result = _flip_keypoints_tensor(x, num_keypoints=NUM_KP)
        assert result.shape == x.shape

    def test_double_flip_identity(self):
        """Flipping twice should return the original."""
        x = torch.randn(1, 8, NUM_KP * 3)
        flipped = _flip_keypoints_tensor(x, num_keypoints=NUM_KP)
        restored = _flip_keypoints_tensor(flipped, num_keypoints=NUM_KP)
        torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)

    def test_x_negated_for_centered(self):
        """For centered coords, x should be negated (before swap)."""
        # Use a single frame with known values
        x = torch.zeros(1, 1, NUM_KP * 3)
        # Set nose (landmark 0, unchanged by swap) x=0.5
        x[0, 0, 0] = 0.5
        result = _flip_keypoints_tensor(x, num_keypoints=NUM_KP)
        # Nose (landmark 0) x should be -0.5
        assert abs(result[0, 0, 0].item() - (-0.5)) < 1e-5


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def _make_model_and_loader(self, num_classes=5, num_samples=20):
        """Create a simple model and DataLoader for testing."""
        class SimpleModel(nn.Module):
            def __init__(self, in_dim, num_classes):
                super().__init__()
                self.fc = nn.Linear(in_dim, num_classes)
            def forward(self, x):
                return self.fc(x.mean(dim=1))

        in_dim = NUM_KP * 3
        model = SimpleModel(in_dim, num_classes)
        model.eval()

        data = torch.randn(num_samples, 16, in_dim)
        labels = torch.randint(0, num_classes, (num_samples,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=4)
        class_names = [f"class_{i}" for i in range(num_classes)]

        return model, loader, class_names

    def test_returns_expected_keys(self):
        model, loader, names = self._make_model_and_loader()
        device = torch.device("cpu")
        result = compute_metrics(model, loader, device, names)
        assert "top1" in result
        assert "top5" in result
        assert "per_class_accuracy" in result
        assert "confusion_matrix" in result
        assert "predictions" in result
        assert "targets" in result

    def test_top1_range(self):
        model, loader, names = self._make_model_and_loader()
        device = torch.device("cpu")
        result = compute_metrics(model, loader, device, names)
        assert 0.0 <= result["top1"] <= 100.0

    def test_confusion_matrix_shape(self):
        model, loader, names = self._make_model_and_loader(num_classes=5)
        device = torch.device("cpu")
        result = compute_metrics(model, loader, device, names)
        assert result["confusion_matrix"].shape == (5, 5)

    def test_per_class_accuracy_keys(self):
        model, loader, names = self._make_model_and_loader(num_classes=3)
        device = torch.device("cpu")
        result = compute_metrics(model, loader, device, names)
        for name in names:
            assert name in result["per_class_accuracy"]

    def test_tta_runs_without_error(self):
        model, loader, names = self._make_model_and_loader(num_classes=5)
        device = torch.device("cpu")
        result = compute_metrics(
            model, loader, device, names,
            use_tta=True, num_keypoints=NUM_KP,
        )
        assert "top1" in result


# ---------------------------------------------------------------------------
# find_hard_negatives
# ---------------------------------------------------------------------------


class TestFindHardNegatives:
    def test_basic(self):
        cm = np.array([
            [10, 3, 0],
            [1, 8, 2],
            [0, 1, 9],
        ])
        names = ["A", "B", "C"]
        pairs = find_hard_negatives(cm, names, top_k=2)
        assert len(pairs) == 2
        # Top confusion: A->B (3 times)
        assert pairs[0] == ("A", "B", 3)

    def test_no_confusions(self):
        cm = np.diag([10, 10, 10])
        names = ["A", "B", "C"]
        pairs = find_hard_negatives(cm, names, top_k=5)
        assert len(pairs) == 0

    def test_top_k_limit(self):
        cm = np.ones((5, 5), dtype=int)
        np.fill_diagonal(cm, 10)
        names = [f"cls_{i}" for i in range(5)]
        pairs = find_hard_negatives(cm, names, top_k=3)
        assert len(pairs) == 3


# ---------------------------------------------------------------------------
# evaluate_latency
# ---------------------------------------------------------------------------


class TestEvaluateLatency:
    def test_returns_expected_keys(self):
        model = nn.Linear(100, 10)
        model.eval()
        result = evaluate_latency(model, torch.device("cpu"), (100,), n_runs=5)
        assert "mean_ms" in result
        assert "std_ms" in result
        assert "fps" in result

    def test_latency_positive(self):
        model = nn.Linear(100, 10)
        model.eval()
        result = evaluate_latency(model, torch.device("cpu"), (100,), n_runs=5)
        assert result["mean_ms"] > 0
        assert result["fps"] > 0


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------


class TestPlotConfusionMatrix:
    def test_creates_file(self, tmp_path):
        cm = np.array([[5, 1], [2, 4]])
        names = ["A", "B"]
        out = tmp_path / "cm.png"
        plot_confusion_matrix(cm, names, out)
        assert out.exists()
        assert out.stat().st_size > 0
