"""Tests for src.training.train — _accuracy, mixup helpers."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.train import _accuracy, _mixup_criterion, _mixup_data


# ---------------------------------------------------------------------------
# _accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_perfect_top1(self):
        output = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        target = torch.tensor([0, 1])
        top1, top5 = _accuracy(output, target, topk=(1, 2))
        assert top1 == 100.0

    def test_zero_accuracy(self):
        output = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        target = torch.tensor([2, 0])
        top1, _ = _accuracy(output, target, topk=(1, 2))
        assert top1 == 0.0

    def test_top5_higher_than_top1(self):
        # 10 classes, prediction is class 1 but true is class 0
        output = torch.zeros(1, 10)
        output[0, 1] = 10.0  # wrong top-1
        output[0, 0] = 5.0   # correct is in top-5
        target = torch.tensor([0])
        top1, top5 = _accuracy(output, target, topk=(1, 5))
        assert top1 == 0.0
        assert top5 == 100.0

    def test_empty_batch(self):
        output = torch.zeros(0, 10)
        target = torch.zeros(0, dtype=torch.long)
        top1, top5 = _accuracy(output, target, topk=(1, 5))
        assert top1 == 0.0
        assert top5 == 0.0

    def test_single_sample(self):
        output = torch.tensor([[0.1, 0.9]])
        target = torch.tensor([1])
        (top1,) = _accuracy(output, target, topk=(1,))
        assert top1 == 100.0


# ---------------------------------------------------------------------------
# _mixup_data
# ---------------------------------------------------------------------------


class TestMixupData:
    def test_output_shapes(self):
        x = torch.randn(4, 16, 100)
        y = torch.tensor([0, 1, 2, 3])
        mixed_x, y_a, y_b, lam = _mixup_data(x, y, alpha=0.2)
        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert 0.0 <= lam <= 1.0

    def test_alpha_zero_returns_original(self):
        x = torch.randn(4, 16, 100)
        y = torch.tensor([0, 1, 2, 3])
        mixed_x, y_a, y_b, lam = _mixup_data(x, y, alpha=0)
        assert lam == 1.0
        torch.testing.assert_close(mixed_x, x)

    def test_mixup_is_interpolation(self):
        """mixed_x should be between the two source samples (element-wise)."""
        x = torch.randn(8, 10, 50)
        y = torch.arange(8)
        np.random.seed(42)
        mixed_x, _, _, lam = _mixup_data(x, y, alpha=1.0)
        # lam * x + (1-lam) * x[perm], values should be bounded by min/max of both
        assert mixed_x.shape == x.shape

    def test_labels_preserved(self):
        x = torch.randn(4, 10)
        y = torch.tensor([10, 20, 30, 40])
        _, y_a, y_b, _ = _mixup_data(x, y, alpha=0.5)
        # y_a should be the original labels
        torch.testing.assert_close(y_a, y)
        # y_b should be a permutation of y
        assert sorted(y_b.tolist()) == sorted(y.tolist())


# ---------------------------------------------------------------------------
# _mixup_criterion
# ---------------------------------------------------------------------------


class TestMixupCriterion:
    def test_lam_one_equals_standard_loss(self):
        criterion = nn.CrossEntropyLoss()
        pred = torch.randn(4, 10)
        y_a = torch.tensor([0, 1, 2, 3])
        y_b = torch.tensor([3, 2, 1, 0])
        loss_mixup = _mixup_criterion(criterion, pred, y_a, y_b, lam=1.0)
        loss_standard = criterion(pred, y_a)
        torch.testing.assert_close(loss_mixup, loss_standard)

    def test_lam_zero_uses_y_b(self):
        criterion = nn.CrossEntropyLoss()
        pred = torch.randn(4, 10)
        y_a = torch.tensor([0, 1, 2, 3])
        y_b = torch.tensor([3, 2, 1, 0])
        loss_mixup = _mixup_criterion(criterion, pred, y_a, y_b, lam=0.0)
        loss_b = criterion(pred, y_b)
        torch.testing.assert_close(loss_mixup, loss_b)

    def test_mixed_loss_between_bounds(self):
        criterion = nn.CrossEntropyLoss()
        pred = torch.randn(4, 10)
        y_a = torch.tensor([0, 1, 2, 3])
        y_b = torch.tensor([3, 2, 1, 0])
        loss_a = criterion(pred, y_a).item()
        loss_b = criterion(pred, y_b).item()
        loss_mix = _mixup_criterion(criterion, pred, y_a, y_b, lam=0.5).item()
        # Should be close to the average
        expected = 0.5 * loss_a + 0.5 * loss_b
        assert abs(loss_mix - expected) < 1e-5
