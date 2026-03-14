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

    def test_mps_tier(self, mps_hw):
        assert auto_config.determine_tier(mps_hw) == "mps"

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
    def test_mid_100(self, cuda_hw):
        cfg = auto_config.build_config_values(100, "mid", cuda_hw)
        assert cfg["approach"] == "stgcn_proto"
        assert cfg["batch_size"] == 32
        assert cfg["T"] == 64
        assert cfg["fp16"] is True
        assert cfg["wlasl_variant"] == 100
        assert cfg["d_model"] == 128
        assert cfg["gcn_channels"] == [64, 128, 128]
        assert cfg["dropout"] == 0.1

    def test_cpu_100(self, cpu_hw):
        cfg = auto_config.build_config_values(100, "cpu", cpu_hw)
        assert cfg["batch_size"] == 8
        assert cfg["T"] == 64
        assert cfg["fp16"] is False
        assert cfg["num_workers"] == min(2, cpu_hw.cpu_cores)

    def test_high_300(self, high_cuda_hw):
        cfg = auto_config.build_config_values(300, "high", high_cuda_hw)
        assert cfg["batch_size"] == 64
        assert cfg["d_model"] == 192
        assert cfg["gcn_channels"] == [64, 128, 256]
        assert cfg["scheduler"] == "cosine"
        assert cfg["epochs"] == 250

    def test_variant_1000(self, cuda_hw):
        cfg = auto_config.build_config_values(1000, "mid", cuda_hw)
        assert cfg["d_model"] == 256
        assert cfg["gcn_channels"] == [64, 128, 256]
        assert cfg["epochs"] == 300

    def test_variant_2000(self, cuda_hw):
        cfg = auto_config.build_config_values(2000, "mid", cuda_hw)
        assert cfg["d_model"] == 384
        assert cfg["gcn_channels"] == [64, 128, 256, 256]
        assert cfg["epochs"] == 350

    def test_num_workers_cuda(self, cuda_hw):
        cfg = auto_config.build_config_values(100, "mid", cuda_hw)
        assert cfg["num_workers"] == min(8, cuda_hw.cpu_cores)

    def test_num_workers_cpu(self, cpu_hw):
        cfg = auto_config.build_config_values(100, "cpu", cpu_hw)
        assert cfg["num_workers"] == min(2, cpu_hw.cpu_cores)

    def test_mps_100(self, mps_hw):
        cfg = auto_config.build_config_values(100, "mps", mps_hw)
        assert cfg["batch_size"] == 16
        assert cfg["fp16"] is False
        assert cfg["num_workers"] == 0
        assert cfg["n_way"] == 10
        assert cfg["num_episodes"] == 200

    def test_buffer_size_matches_t(self, cuda_hw):
        cfg = auto_config.build_config_values(100, "mid", cuda_hw)
        assert cfg["buffer_size"] == cfg["T"]

    def test_prototypical_fields(self, cuda_hw):
        cfg = auto_config.build_config_values(100, "mid", cuda_hw)
        assert "n_way" in cfg
        assert "k_shot" in cfg
        assert "q_query" in cfg
        assert "num_episodes" in cfg


# ---------------------------------------------------------------------------
# TestRenderYaml
# ---------------------------------------------------------------------------


class TestRenderYaml:
    def test_valid_yaml(self, cuda_hw):
        values = auto_config.build_config_values(100, "mid", cuda_hw)
        content = auto_config.render_yaml(values, cuda_hw, "mid")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert parsed["approach"] == "stgcn_proto"

    def test_header_present(self, cuda_hw):
        values = auto_config.build_config_values(100, "mid", cuda_hw)
        content = auto_config.render_yaml(values, cuda_hw, "mid")
        assert "Auto-generated by scripts/auto_config.py" in content
        assert "NVIDIA RTX 3070" in content
        assert "Tier: mid" in content

    def test_fp16_false_in_cpu_yaml(self, cpu_hw):
        values = auto_config.build_config_values(100, "cpu", cpu_hw)
        content = auto_config.render_yaml(values, cpu_hw, "cpu")
        parsed = yaml.safe_load(content)
        assert parsed["fp16"] is False

    def test_gcn_channels_in_yaml(self, cuda_hw):
        values = auto_config.build_config_values(100, "mid", cuda_hw)
        content = auto_config.render_yaml(values, cuda_hw, "mid")
        parsed = yaml.safe_load(content)
        assert parsed["gcn_channels"] == [64, 128, 128]


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
            [sys.executable, "scripts/auto_config.py", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "WLASL Auto-Config" in result.stdout
        assert "approach: stgcn_proto" in result.stdout

    def test_dry_run_variant_300(self):
        result = subprocess.run(
            [sys.executable, "scripts/auto_config.py",
             "--variant", "300", "--device", "cpu", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "wlasl_variant: 300" in result.stdout

    def test_write_and_load(self, tmp_path):
        out = tmp_path / "test.yaml"
        result = subprocess.run(
            [sys.executable, "scripts/auto_config.py",
             "--output", str(out)],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert parsed["approach"] == "stgcn_proto"
