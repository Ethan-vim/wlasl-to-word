"""Tests for src.inference.export_onnx — ONNX export and verification."""

import pytest
import torch

from src.models.pose_transformer import PoseTransformer
from src.training.config import Config

NUM_KP = 543


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


class TestExportOnnx:
    def _make_model_and_cfg(self, use_motion=False):
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
        model.eval()
        return model, cfg

    def test_export_creates_file(self, tmp_path):
        from src.inference.export_onnx import export_to_onnx

        model, cfg = self._make_model_and_cfg()
        out = tmp_path / "model.onnx"
        result = export_to_onnx(model, cfg, out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_with_motion(self, tmp_path):
        from src.inference.export_onnx import export_to_onnx

        model, cfg = self._make_model_and_cfg(use_motion=True)
        out = tmp_path / "model_motion.onnx"
        result = export_to_onnx(model, cfg, out)
        assert result.exists()

    @pytest.mark.skipif(
        not _onnxruntime_available(),
        reason="onnxruntime not installed",
    )
    def test_verify_onnx(self, tmp_path):
        from src.inference.export_onnx import export_to_onnx, verify_onnx

        model, cfg = self._make_model_and_cfg()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, cfg, out)
        assert verify_onnx(out, cfg) is True

    @pytest.mark.skipif(
        not _onnxruntime_available(),
        reason="onnxruntime not installed",
    )
    def test_verify_onnx_with_motion(self, tmp_path):
        from src.inference.export_onnx import export_to_onnx, verify_onnx

        model, cfg = self._make_model_and_cfg(use_motion=True)
        out = tmp_path / "model.onnx"
        export_to_onnx(model, cfg, out)
        assert verify_onnx(out, cfg) is True
