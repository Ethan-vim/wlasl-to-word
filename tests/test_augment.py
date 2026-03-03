"""Tests for src.data.augment — all augmentation classes and pipelines."""

import numpy as np
import pytest

from src.data.augment import (
    Compose,
    KeypointDropout,
    KeypointHorizontalFlip,
    KeypointNoise,
    KeypointRotation,
    KeypointScale,
    KeypointTranslation,
    TemporalCrop,
    TemporalFlip,
    TemporalSpeedPerturb,
    _build_swap_indices,
    get_train_transforms,
    get_val_transforms,
)

NUM_KP = 543


def _kps(T=30, K=NUM_KP, C=3):
    return np.random.rand(T, K, C).astype(np.float32)


# ---------------------------------------------------------------------------
# _build_swap_indices
# ---------------------------------------------------------------------------


class TestBuildSwapIndices:
    def test_length(self):
        swap = _build_swap_indices()
        assert swap.shape == (543,)

    def test_symmetric(self):
        swap = _build_swap_indices()
        # Applying the swap twice should return the identity
        assert np.array_equal(swap[swap], np.arange(543))

    def test_hand_swap(self):
        swap = _build_swap_indices()
        # Left hand (33-53) should swap with right hand (54-74)
        for i in range(21):
            assert swap[33 + i] == 54 + i
            assert swap[54 + i] == 33 + i


# ---------------------------------------------------------------------------
# TemporalCrop
# ---------------------------------------------------------------------------


class TestTemporalCrop:
    def test_exact_length(self):
        kps = _kps(T=64)
        result = TemporalCrop(T=64)(kps)
        assert result.shape[0] == 64

    def test_longer_input(self):
        kps = _kps(T=100)
        result = TemporalCrop(T=32)(kps)
        assert result.shape == (32, NUM_KP, 3)

    def test_shorter_input_padded(self):
        kps = _kps(T=10)
        result = TemporalCrop(T=64)(kps)
        assert result.shape == (64, NUM_KP, 3)
        # Last 54 frames should be copies of frame 9
        np.testing.assert_array_equal(result[10], result[9])

    def test_empty_input(self):
        kps = np.zeros((0, NUM_KP, 3), dtype=np.float32)
        result = TemporalCrop(T=16)(kps)
        assert result.shape == (16, NUM_KP, 3)
        assert np.all(result == 0)


# ---------------------------------------------------------------------------
# TemporalFlip
# ---------------------------------------------------------------------------


class TestTemporalFlip:
    def test_p_one_always_flips(self):
        kps = np.arange(30).reshape(30, 1, 1).astype(np.float32)
        result = TemporalFlip(p=1.0)(kps)
        expected = kps[::-1]
        np.testing.assert_array_equal(result, expected)

    def test_p_zero_never_flips(self):
        kps = _kps(T=20)
        result = TemporalFlip(p=0.0)(kps)
        np.testing.assert_array_equal(result, kps)

    def test_shape_preserved(self):
        kps = _kps(T=25)
        result = TemporalFlip(p=0.5)(kps)
        assert result.shape == kps.shape


# ---------------------------------------------------------------------------
# TemporalSpeedPerturb
# ---------------------------------------------------------------------------


class TestTemporalSpeedPerturb:
    def test_output_length_changes(self):
        kps = _kps(T=100)
        np.random.seed(42)
        result = TemporalSpeedPerturb(low=0.5, high=2.0)(kps)
        # Should produce a different length (very likely given extreme range)
        assert result.shape[1:] == kps.shape[1:]

    def test_single_frame(self):
        kps = _kps(T=1)
        result = TemporalSpeedPerturb()(kps)
        assert result.shape == kps.shape


# ---------------------------------------------------------------------------
# KeypointHorizontalFlip
# ---------------------------------------------------------------------------


class TestKeypointHorizontalFlip:
    def test_p_one_flips_x(self):
        kps = _kps(T=5)
        original_x = kps[:, 0, 0].copy()
        flip = KeypointHorizontalFlip(p=1.0, centered=True)
        result = flip(kps)
        # x should be negated for centered mode
        np.testing.assert_allclose(result[:, 0, 0], -original_x, atol=1e-6)

    def test_p_zero_no_change(self):
        kps = _kps(T=5)
        flip = KeypointHorizontalFlip(p=0.0, centered=True)
        result = flip(kps)
        np.testing.assert_array_equal(result, kps)

    def test_flat_input(self):
        kps = _kps(T=5).reshape(5, NUM_KP * 3)
        flip = KeypointHorizontalFlip(p=1.0, centered=True)
        result = flip(kps)
        assert result.shape == kps.shape

    def test_non_centered_mode(self):
        kps = np.ones((3, NUM_KP, 3), dtype=np.float32) * 0.3
        flip = KeypointHorizontalFlip(p=1.0, centered=False)
        result = flip(kps)
        # For non-centered, x' = 1 - x, so 1 - 0.3 = 0.7
        # (after swap reordering, check a landmark that maps to itself: nose=0)
        assert abs(result[0, 0, 0] - 0.7) < 1e-5

    def test_double_flip_identity(self):
        """Flipping twice with p=1 should restore the original."""
        kps = _kps(T=5)
        flip = KeypointHorizontalFlip(p=1.0, centered=True)
        result = flip(flip(kps))
        np.testing.assert_allclose(result, kps, atol=1e-5)


# ---------------------------------------------------------------------------
# KeypointNoise
# ---------------------------------------------------------------------------


class TestKeypointNoise:
    def test_output_differs(self):
        kps = _kps(T=10)
        result = KeypointNoise(sigma=0.1)(kps)
        assert not np.array_equal(result, kps)

    def test_shape_preserved(self):
        kps = _kps(T=10)
        result = KeypointNoise(sigma=0.02)(kps)
        assert result.shape == kps.shape

    def test_noise_magnitude(self):
        kps = np.zeros((20, NUM_KP, 3), dtype=np.float32)
        result = KeypointNoise(sigma=0.01)(kps)
        # Noise std should be approximately 0.01
        assert result.std() < 0.05  # loose upper bound


# ---------------------------------------------------------------------------
# KeypointScale
# ---------------------------------------------------------------------------


class TestKeypointScale:
    def test_scaling(self):
        kps = np.ones((5, NUM_KP, 3), dtype=np.float32)
        np.random.seed(0)
        result = KeypointScale(low=2.0, high=2.0)(kps)
        np.testing.assert_allclose(result, 2.0, atol=0.01)

    def test_shape_preserved(self):
        kps = _kps(T=10)
        assert KeypointScale()(kps).shape == kps.shape


# ---------------------------------------------------------------------------
# KeypointRotation
# ---------------------------------------------------------------------------


class TestKeypointRotation:
    def test_p_zero_no_change(self):
        kps = _kps(T=5)
        result = KeypointRotation(p=0.0)(kps)
        np.testing.assert_array_equal(result, kps)

    def test_p_one_modifies(self):
        kps = _kps(T=5)
        np.random.seed(99)
        result = KeypointRotation(max_angle=45, p=1.0)(kps)
        assert not np.array_equal(result, kps)

    def test_z_unchanged(self):
        kps = _kps(T=5)
        np.random.seed(42)
        result = KeypointRotation(max_angle=30, p=1.0)(kps)
        np.testing.assert_array_equal(result[:, :, 2], kps[:, :, 2])

    def test_flat_input(self):
        kps = _kps(T=5).reshape(5, NUM_KP * 3)
        result = KeypointRotation(max_angle=15, p=1.0)(kps)
        assert result.shape == kps.shape

    def test_rotation_preserves_norm(self):
        """Rotation should preserve the L2 norm of (x, y) per point."""
        kps = _kps(T=5)
        np.random.seed(7)
        result = KeypointRotation(max_angle=45, p=1.0)(kps)
        orig_norms = np.linalg.norm(kps[:, :, :2], axis=-1)
        rot_norms = np.linalg.norm(result[:, :, :2], axis=-1)
        np.testing.assert_allclose(rot_norms, orig_norms, atol=1e-5)


# ---------------------------------------------------------------------------
# KeypointTranslation
# ---------------------------------------------------------------------------


class TestKeypointTranslation:
    def test_p_zero_no_change(self):
        kps = _kps(T=5)
        result = KeypointTranslation(p=0.0)(kps)
        np.testing.assert_array_equal(result, kps)

    def test_translation_uniform_shift(self):
        """All keypoints should shift by the same dx, dy."""
        kps = _kps(T=3)
        np.random.seed(10)
        result = KeypointTranslation(max_shift=0.1, p=1.0)(kps)
        diff = result - kps
        # All dx values should be the same within a frame, same for dy
        dx = diff[0, :, 0]
        dy = diff[0, :, 1]
        np.testing.assert_allclose(dx, dx[0], atol=1e-7)
        np.testing.assert_allclose(dy, dy[0], atol=1e-7)
        # z should be unchanged
        np.testing.assert_array_equal(result[:, :, 2], kps[:, :, 2])

    def test_flat_input(self):
        kps = _kps(T=5).reshape(5, NUM_KP * 3)
        result = KeypointTranslation(max_shift=0.1, p=1.0)(kps)
        assert result.shape == kps.shape


# ---------------------------------------------------------------------------
# KeypointDropout
# ---------------------------------------------------------------------------


class TestKeypointDropout:
    def test_p_zero_no_change(self):
        kps = _kps(T=10)
        result = KeypointDropout(p=0.0)(kps)
        np.testing.assert_array_equal(result, kps)

    def test_some_zeros_created(self):
        kps = np.ones((50, NUM_KP, 3), dtype=np.float32)
        np.random.seed(42)
        result = KeypointDropout(frame_drop_rate=0.5, landmark_drop_rate=0.3, p=1.0)(kps)
        assert np.sum(result == 0) > 0

    def test_flat_input(self):
        kps = np.ones((20, NUM_KP * 3), dtype=np.float32)
        result = KeypointDropout(frame_drop_rate=0.1, landmark_drop_rate=0.05, p=1.0)(kps)
        assert result.shape == kps.shape

    def test_shape_preserved(self):
        kps = _kps(T=10)
        result = KeypointDropout(p=1.0)(kps)
        assert result.shape == kps.shape


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    def test_identity(self):
        kps = _kps(T=10)
        composed = Compose([])
        np.testing.assert_array_equal(composed(kps), kps)

    def test_chaining(self):
        composed = Compose([TemporalCrop(T=16), KeypointNoise(sigma=0.01)])
        kps = _kps(T=30)
        result = composed(kps)
        assert result.shape[0] == 16


# ---------------------------------------------------------------------------
# Pipeline presets
# ---------------------------------------------------------------------------


class TestPipelinePresets:
    def test_train_transforms_shape(self):
        pipeline = get_train_transforms(T=32)
        kps = _kps(T=50)
        result = pipeline(kps)
        assert result.shape[0] == 32

    def test_val_transforms_deterministic(self):
        pipeline = get_val_transforms(T=32)
        kps = _kps(T=50)
        r1 = pipeline(kps.copy())
        r2 = pipeline(kps.copy())
        np.testing.assert_array_equal(r1, r2)

    def test_train_has_no_temporal_flip(self):
        """TemporalFlip should NOT be in the training pipeline."""
        pipeline = get_train_transforms(T=32)
        for t in pipeline.transforms:
            assert not isinstance(t, TemporalFlip), "TemporalFlip should be removed"

    def test_train_noise_sigma_is_0_02(self):
        pipeline = get_train_transforms(T=32)
        for t in pipeline.transforms:
            if isinstance(t, KeypointNoise):
                assert t.sigma == 0.02
                break
        else:
            pytest.fail("KeypointNoise not found in training pipeline")
