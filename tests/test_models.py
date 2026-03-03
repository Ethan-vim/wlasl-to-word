"""Tests for src.models — PoseTransformer, PoseBiLSTM, FusionModel, build functions."""

import pytest
import torch

from src.models.pose_transformer import (
    PoseBiLSTM,
    PoseTransformer,
    build_pose_model,
)
from src.training.config import Config


NUM_KP = 543
B, T = 2, 16


# ---------------------------------------------------------------------------
# PoseTransformer
# ---------------------------------------------------------------------------


class TestPoseTransformer:
    def test_forward_shape(self):
        model = PoseTransformer(
            num_keypoints=NUM_KP, num_classes=10, d_model=64, nhead=4,
            num_layers=2, T=T, use_motion=False,
        )
        x = torch.randn(B, T, NUM_KP * 3)
        out = model(x)
        assert out.shape == (B, 10)

    def test_forward_with_motion(self):
        model = PoseTransformer(
            num_keypoints=NUM_KP, num_classes=10, d_model=64, nhead=4,
            num_layers=2, T=T, use_motion=True,
        )
        x = torch.randn(B, T, NUM_KP * 6)
        out = model(x)
        assert out.shape == (B, 10)

    def test_get_features(self):
        model = PoseTransformer(
            num_keypoints=NUM_KP, num_classes=10, d_model=64, nhead=4,
            num_layers=2, T=T,
        )
        x = torch.randn(B, T, NUM_KP * 3)
        feat = model.get_features(x)
        assert feat.shape == (B, 64)

    def test_variable_seq_length(self):
        model = PoseTransformer(
            num_keypoints=NUM_KP, num_classes=10, d_model=64, nhead=4,
            num_layers=2, T=32,
        )
        # Input shorter than T should still work
        x = torch.randn(B, 8, NUM_KP * 3)
        out = model(x)
        assert out.shape == (B, 10)


# ---------------------------------------------------------------------------
# PoseBiLSTM
# ---------------------------------------------------------------------------


class TestPoseBiLSTM:
    def test_forward_shape(self):
        model = PoseBiLSTM(
            num_keypoints=NUM_KP, num_classes=10, d_model=64,
            num_layers=2, T=T, use_motion=False,
        )
        x = torch.randn(B, T, NUM_KP * 3)
        out = model(x)
        assert out.shape == (B, 10)

    def test_forward_with_motion(self):
        model = PoseBiLSTM(
            num_keypoints=NUM_KP, num_classes=10, d_model=64,
            num_layers=2, T=T, use_motion=True,
        )
        x = torch.randn(B, T, NUM_KP * 6)
        out = model(x)
        assert out.shape == (B, 10)

    def test_get_features(self):
        model = PoseBiLSTM(
            num_keypoints=NUM_KP, num_classes=10, d_model=64,
            num_layers=2, T=T,
        )
        x = torch.randn(B, T, NUM_KP * 3)
        feat = model.get_features(x)
        assert feat.shape == (B, 64)


# ---------------------------------------------------------------------------
# build_pose_model
# ---------------------------------------------------------------------------


class TestBuildPoseModel:
    def test_build_transformer(self):
        cfg = Config(
            approach="pose_transformer", num_keypoints=NUM_KP, num_classes=10,
            wlasl_variant=10,
            d_model=64, nhead=4, num_layers=2, T=T, use_motion=False,
        )
        model = build_pose_model(cfg)
        assert isinstance(model, PoseTransformer)
        x = torch.randn(1, T, NUM_KP * 3)
        assert model(x).shape == (1, 10)

    def test_build_bilstm(self):
        cfg = Config(
            approach="pose_bilstm", num_keypoints=NUM_KP, num_classes=10,
            wlasl_variant=10,
            d_model=64, num_layers=2, T=T, use_motion=False,
        )
        model = build_pose_model(cfg)
        assert isinstance(model, PoseBiLSTM)

    def test_build_with_motion(self):
        cfg = Config(
            approach="pose_transformer", num_keypoints=NUM_KP, num_classes=10,
            wlasl_variant=10,
            d_model=64, nhead=4, num_layers=2, T=T, use_motion=True,
        )
        model = build_pose_model(cfg)
        x = torch.randn(1, T, NUM_KP * 6)
        assert model(x).shape == (1, 10)

    def test_unknown_approach_raises(self):
        cfg = Config(approach="unknown_model")
        with pytest.raises(ValueError, match="Unknown pose approach"):
            build_pose_model(cfg)

    def test_model_is_differentiable(self):
        cfg = Config(
            approach="pose_transformer", num_keypoints=NUM_KP, num_classes=10,
            wlasl_variant=10,
            d_model=64, nhead=4, num_layers=2, T=T, use_motion=False,
        )
        model = build_pose_model(cfg)
        x = torch.randn(1, T, NUM_KP * 3)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check at least one parameter has gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


# ---------------------------------------------------------------------------
# FusionModel (basic smoke test, avoids heavy video backbone download)
# ---------------------------------------------------------------------------


class TestFusionModel:
    def test_concat_fusion(self):
        from src.models.fusion import CrossAttentionFusion, FusionModel

        # Build minimal pose model
        pose = PoseTransformer(
            num_keypoints=NUM_KP, num_classes=10, d_model=32,
            nhead=4, num_layers=1, T=T,
        )
        # Mock a simple video model with get_features
        class DummyVideoModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feat_dim = 64
                self.fc = torch.nn.Linear(64, 64)
            def get_features(self, x):
                return torch.randn(x.size(0), self.feat_dim)
            def forward(self, x):
                return self.fc(self.get_features(x))

        video = DummyVideoModel()
        fusion = FusionModel(pose, video, num_classes=10, fusion="concat")
        pose_input = torch.randn(B, T, NUM_KP * 3)
        video_input = torch.randn(B, 3, T, 112, 112)
        out = fusion(pose_input, video_input)
        assert out.shape == (B, 10)

    def test_attention_fusion(self):
        from src.models.fusion import FusionModel

        pose = PoseTransformer(
            num_keypoints=NUM_KP, num_classes=10, d_model=32,
            nhead=4, num_layers=1, T=T,
        )
        class DummyVideoModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feat_dim = 64
            def get_features(self, x):
                return torch.randn(x.size(0), self.feat_dim)
            def forward(self, x):
                return self.get_features(x)

        video = DummyVideoModel()
        fusion = FusionModel(pose, video, num_classes=10, fusion="attention", fusion_dim=32)
        pose_input = torch.randn(B, T, NUM_KP * 3)
        video_input = torch.randn(B, 3, T, 112, 112)
        out = fusion(pose_input, video_input)
        assert out.shape == (B, 10)
