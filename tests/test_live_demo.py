"""Tests for src.inference.live_demo — FrameBuffer, prediction smoothing."""

import numpy as np
import pytest

from src.inference.live_demo import FrameBuffer, LivePredictor

NUM_KP = 543


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------


class TestFrameBuffer:
    def test_push_and_len(self):
        buf = FrameBuffer(max_size=10)
        assert len(buf) == 0
        frame = np.random.rand(NUM_KP, 3).astype(np.float32)
        buf.push(frame)
        assert len(buf) == 1

    def test_max_size(self):
        buf = FrameBuffer(max_size=5)
        for _ in range(10):
            buf.push(np.random.rand(NUM_KP, 3).astype(np.float32))
        assert len(buf) == 5

    def test_get_all_shape(self):
        buf = FrameBuffer(max_size=10)
        for _ in range(7):
            buf.push(np.random.rand(NUM_KP, 3).astype(np.float32))
        result = buf.get_all()
        assert result.shape == (7, NUM_KP, 3)

    def test_get_all_empty(self):
        buf = FrameBuffer(max_size=10)
        result = buf.get_all()
        assert result.shape == (0, NUM_KP, 3)

    def test_clear(self):
        buf = FrameBuffer(max_size=10)
        for _ in range(5):
            buf.push(np.random.rand(NUM_KP, 3).astype(np.float32))
        buf.clear()
        assert len(buf) == 0
        assert buf.get_all().shape[0] == 0

    def test_fifo_order(self):
        """Oldest frames should be dropped first."""
        buf = FrameBuffer(max_size=3)
        for i in range(5):
            frame = np.full((NUM_KP, 3), float(i), dtype=np.float32)
            buf.push(frame)
        result = buf.get_all()
        # Should contain frames 2, 3, 4
        assert result[0, 0, 0] == 2.0
        assert result[-1, 0, 0] == 4.0


# ---------------------------------------------------------------------------
# LivePredictor.smooth_predictions (static method)
# ---------------------------------------------------------------------------


class TestSmoothPredictions:
    def _pred(self, gloss, confidence, label_idx=0):
        return {
            "gloss": gloss,
            "confidence": confidence,
            "label_idx": label_idx,
            "top5": [(gloss, confidence)],
        }

    def test_empty_returns_none(self):
        result = LivePredictor.smooth_predictions([], mode="avg")
        assert result is None

    def test_majority_mode(self):
        preds = [
            self._pred("hello", 0.9, 0),
            self._pred("hello", 0.8, 0),
            self._pred("world", 0.7, 1),
        ]
        result = LivePredictor.smooth_predictions(preds, mode="majority")
        assert result["gloss"] == "hello"

    def test_avg_mode_picks_highest(self):
        preds = [
            self._pred("hello", 0.9, 0),
            self._pred("hello", 0.8, 0),
            self._pred("world", 0.7, 1),
        ]
        result = LivePredictor.smooth_predictions(preds, mode="avg")
        # "hello" has total prob 1.7 (avg 0.567) vs "world" 0.7 (avg 0.233)
        assert result["gloss"] == "hello"

    def test_single_prediction(self):
        preds = [self._pred("test", 0.95, 5)]
        result = LivePredictor.smooth_predictions(preds, mode="avg")
        assert result["gloss"] == "test"
        assert abs(result["confidence"] - 0.95) < 1e-5
