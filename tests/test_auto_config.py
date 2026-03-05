"""Tests for scripts/auto_config.py — hardware detection and config generation."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import auto_config  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cuda_hw():
    """Simulated CUDA hardware (mid-tier GPU)."""
    return auto_config.HardwareInfo(
        device="cuda",
        device_name="NVIDIA RTX 3070",
        vram_gb=8.0,
        cuda_version="12.1",
        cpu_cores=8,
        platform_name="Linux-6.5.0-x86_64",
        torch_version="2.2.0",
        gpu_count=1,
    )


@pytest.fixture
def high_cuda_hw():
    """Simulated high-tier CUDA hardware."""
    return auto_config.HardwareInfo(
        device="cuda",
        device_name="NVIDIA A100",
        vram_gb=40.0,
        cuda_version="12.1",
        cpu_cores=32,
        platform_name="Linux-6.5.0-x86_64",
        torch_version="2.2.0",
        gpu_count=4,
    )


@pytest.fixture
def low_cuda_hw():
    """Simulated low-tier CUDA hardware."""
    return auto_config.HardwareInfo(
        device="cuda",
        device_name="NVIDIA GTX 1650",
        vram_gb=4.0,
        cuda_version="11.8",
        cpu_cores=4,
        platform_name="Windows-10-x86_64",
        torch_version="2.1.0",
        gpu_count=1,
    )


@pytest.fixture
def mps_hw():
    """Simulated Apple Silicon hardware."""
    return auto_config.HardwareInfo(
        device="mps",
        device_name="Apple arm64",
        vram_gb=0.0,
        cuda_version="",
        cpu_cores=10,
        platform_name="macOS-14.0-arm64",
        torch_version="2.2.0",
        gpu_count=0,
    )


@pytest.fixture
def cpu_hw():
    """Simulated CPU-only hardware."""
    return auto_config.HardwareInfo(
        device="cpu",
        device_name="Intel i7",
        vram_gb=0.0,
        cuda_version="",
        cpu_cores=8,
        platform_name="Linux-6.5.0-x86_64",
        torch_version="2.2.0",
        gpu_count=0,
    )


# ---------------------------------------------------------------------------
# TestDetermineTier
# ---------------------------------------------------------------------------


class TestDetermineTier:
    def test_high_tier(self, high_cuda_hw):
        assert auto_config.determine_tier(high_cuda_hw) == "high"

    def test_mid_tier(self, cuda_hw):
        assert auto_config.determine_tier(cuda_hw) == "mid"

    def test_low_tier(self, low_cuda_hw):
        assert auto_config.determine_tier(low_cuda_hw) == "low"

    def test_mps_is_cpu_tier(self, mps_hw):
        assert auto_config.determine_tier(mps_hw) == "cpu"

    def test_cpu_is_cpu_tier(self, cpu_hw):
        assert auto_config.determine_tier(cpu_hw) == "cpu"

    def test_boundary_16gb(self):
        hw = auto_config.HardwareInfo(
            device="cuda", device_name="GPU", vram_gb=16.0,
            cuda_version="12.1", cpu_cores=8, platform_name="Linux",
            torch_version="2.2.0", gpu_count=1,
        )
        assert auto_config.determine_tier(hw) == "high"

    def test_boundary_8gb(self):
        hw = auto_config.HardwareInfo(
            device="cuda", device_name="GPU", vram_gb=8.0,
            cuda_version="12.1", cpu_cores=8, platform_name="Linux",
            torch_version="2.2.0", gpu_count=1,
        )
        assert auto_config.determine_tier(hw) == "mid"

    def test_boundary_below_4gb(self):
        hw = auto_config.HardwareInfo(
            device="cuda", device_name="GPU", vram_gb=3.5,
            cuda_version="11.8", cpu_cores=4, platform_name="Linux",
            torch_version="2.1.0", gpu_count=1,
        )
        assert auto_config.determine_tier(hw) == "low"


# ---------------------------------------------------------------------------
# TestBuildConfigValues
# ---------------------------------------------------------------------------


class TestBuildConfigValues:
    def test_pose_mid_100(self, cuda_hw):
        cfg = auto_config.build_config_values("pose", 100, "mid", cuda_hw)
        assert cfg["approach"] == "pose_transformer"
        assert cfg["batch_size"] == 32
        assert cfg["T"] == 64
        assert cfg["fp16"] is True
        assert cfg["wlasl_variant"] == 100
        assert cfg["d_model"] == 256
        assert cfg["num_layers"] == 4

    def test_pose_cpu_100(self, cpu_hw):
        cfg = auto_config.build_config_values("pose", 100, "cpu", cpu_hw)
        assert cfg["batch_size"] == 8
        assert cfg["T"] == 32
        assert cfg["fp16"] is False
        assert cfg["num_workers"] == 2

    def test_pose_high_300(self, high_cuda_hw):
        cfg = auto_config.build_config_values("pose", 300, "high", high_cuda_hw)
        assert cfg["batch_size"] == 64
        assert cfg["num_layers"] == 6
        assert cfg["scheduler"] == "cosine"
        assert cfg["epochs"] == 150

    def test_pose_1000(self, cuda_hw):
        cfg = auto_config.build_config_values("pose", 1000, "mid", cuda_hw)
        assert cfg["d_model"] == 384
        assert cfg["num_layers"] == 6
        assert cfg["epochs"] == 200

    def test_video_mid_100(self, cuda_hw):
        cfg = auto_config.build_config_values("video", 100, "mid", cuda_hw)
        assert cfg["approach"] == "video"
        assert cfg["batch_size"] == 8
        assert cfg["T"] == 32
        assert cfg["image_size"] == 224
        assert cfg["fp16"] is True

    def test_video_low_100(self, low_cuda_hw):
        cfg = auto_config.build_config_values("video", 100, "low", low_cuda_hw)
        assert cfg["batch_size"] == 4
        assert cfg["T"] == 16
        assert cfg["image_size"] == 112

    def test_fusion_cpu(self, cpu_hw):
        cfg = auto_config.build_config_values("fusion", 100, "cpu", cpu_hw)
        assert cfg["approach"] == "fusion"
        assert cfg["batch_size"] == 4
        assert cfg["T"] == 32
        assert cfg["image_size"] == 112
        assert cfg["fp16"] is False

    def test_num_workers_cuda(self, cuda_hw):
        cfg = auto_config.build_config_values("pose", 100, "mid", cuda_hw)
        assert cfg["num_workers"] == min(8, cuda_hw.cpu_cores)

    def test_num_workers_cpu(self, cpu_hw):
        cfg = auto_config.build_config_values("pose", 100, "cpu", cpu_hw)
        assert cfg["num_workers"] == min(2, cpu_hw.cpu_cores)

    def test_buffer_size_matches_t(self, cuda_hw):
        cfg = auto_config.build_config_values("pose", 100, "mid", cuda_hw)
        assert cfg["buffer_size"] == cfg["T"]

    def test_variant_300_video_epochs(self, cuda_hw):
        cfg = auto_config.build_config_values("video", 300, "mid", cuda_hw)
        assert cfg["epochs"] == 150

    def test_variant_2000_fusion_epochs(self, cuda_hw):
        cfg = auto_config.build_config_values("fusion", 2000, "mid", cuda_hw)
        assert cfg["epochs"] == 200
        assert cfg["early_stopping_patience"] == 30


# ---------------------------------------------------------------------------
# TestRenderYaml
# ---------------------------------------------------------------------------


class TestRenderYaml:
    def test_valid_yaml(self, cuda_hw):
        values = auto_config.build_config_values("pose", 100, "mid", cuda_hw)
        content = auto_config.render_yaml("pose", values, cuda_hw, "mid")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert parsed["approach"] == "pose_transformer"

    def test_header_present(self, cuda_hw):
        values = auto_config.build_config_values("pose", 100, "mid", cuda_hw)
        content = auto_config.render_yaml("pose", values, cuda_hw, "mid")
        assert "Auto-generated by scripts/auto_config.py" in content
        assert "NVIDIA RTX 3070" in content
        assert "Tier: mid" in content

    def test_video_yaml_valid(self, cuda_hw):
        values = auto_config.build_config_values("video", 100, "mid", cuda_hw)
        content = auto_config.render_yaml("video", values, cuda_hw, "mid")
        parsed = yaml.safe_load(content)
        assert parsed["backbone"] == "r2plus1d_18"

    def test_fusion_yaml_valid(self, cuda_hw):
        values = auto_config.build_config_values("fusion", 100, "mid", cuda_hw)
        content = auto_config.render_yaml("fusion", values, cuda_hw, "mid")
        parsed = yaml.safe_load(content)
        assert parsed["fusion"] == "concat"
        assert parsed["fusion_dim"] == 256

    def test_fp16_false_in_cpu_yaml(self, cpu_hw):
        values = auto_config.build_config_values("pose", 100, "cpu", cpu_hw)
        content = auto_config.render_yaml("pose", values, cpu_hw, "cpu")
        parsed = yaml.safe_load(content)
        assert parsed["fp16"] is False


# ---------------------------------------------------------------------------
# TestDetectHardware
# ---------------------------------------------------------------------------


class TestDetectHardware:
    def test_device_override_cpu(self):
        hw = auto_config.detect_hardware(device_override="cpu")
        assert hw.device == "cpu"
        assert hw.cpu_cores > 0
        assert hw.torch_version != ""

    def test_returns_hardware_info(self):
        hw = auto_config.detect_hardware()
        assert isinstance(hw, auto_config.HardwareInfo)
        assert hw.device in ("cuda", "mps", "cpu")
        assert hw.cpu_cores > 0


# ---------------------------------------------------------------------------
# TestMainDryRun (integration)
# ---------------------------------------------------------------------------


class TestMainDryRun:
    def test_dry_run_succeeds(self):
        result = subprocess.run(
            [sys.executable, "scripts/auto_config.py", "--approach", "pose", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "WLASL Auto-Config" in result.stdout
        assert "approach: pose_transformer" in result.stdout

    def test_dry_run_video(self):
        result = subprocess.run(
            [sys.executable, "scripts/auto_config.py",
             "--approach", "video", "--variant", "300", "--device", "cpu", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "approach: video" in result.stdout
        assert "wlasl_variant: 300" in result.stdout

    def test_dry_run_fusion(self):
        result = subprocess.run(
            [sys.executable, "scripts/auto_config.py",
             "--approach", "fusion", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "approach: fusion" in result.stdout
        assert "fusion: concat" in result.stdout

    def test_write_and_load(self, tmp_path):
        out = tmp_path / "test.yaml"
        result = subprocess.run(
            [sys.executable, "scripts/auto_config.py",
             "--approach", "pose", "--output", str(out)],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert parsed["approach"] == "pose_transformer"
