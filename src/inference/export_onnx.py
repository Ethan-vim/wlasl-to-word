"""
ONNX export, verification, and benchmarking for trained models.

Exports a PyTorch model to ONNX format, verifies the exported model's
structure and output shape via ONNX Runtime, and benchmarks inference
latency.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from src.models.pose_transformer import build_pose_model
from src.models.video_i3d import build_video_model
from src.training.config import Config, load_config

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    cfg: Config,
    output_path: str | Path,
    opset: int = 17,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Parameters
    ----------
    model : nn.Module
        Trained model (already in eval mode on CPU).
    cfg : Config
        Configuration (used to determine input shape).
    output_path : str or Path
        Destination ``.onnx`` file.
    opset : int
        ONNX opset version.

    Returns
    -------
    Path
        Path to the saved ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.cpu()

    # Create dummy input
    if cfg.approach in ("pose_transformer", "pose_bilstm"):
        features_per_kp = 6 if getattr(cfg, "use_motion", False) else 3
        dummy_input = torch.randn(1, cfg.T, cfg.num_keypoints * features_per_kp)
        input_names = ["keypoints"]
        dynamic_axes = {"keypoints": {0: "batch_size"}, "logits": {0: "batch_size"}}
    elif cfg.approach == "video":
        dummy_input = torch.randn(1, 3, cfg.T, cfg.image_size, cfg.image_size)
        input_names = ["video"]
        dynamic_axes = {"video": {0: "batch_size"}, "logits": {0: "batch_size"}}
    else:
        raise ValueError(f"ONNX export not supported for approach '{cfg.approach}'")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    logger.info("Exported ONNX model to %s", output_path)
    return output_path


def verify_onnx(
    onnx_path: str | Path,
    cfg: Config,
    atol: float = 1e-4,
) -> bool:
    """Verify an ONNX model's structure and output shape.

    Validates the ONNX model with ``onnx.checker``, runs a random input
    through ONNX Runtime, and asserts the output shape matches
    ``(1, num_classes)``.

    Parameters
    ----------
    onnx_path : str or Path
        Path to the ONNX model.
    cfg : Config
        Configuration.
    atol : float
        Reserved for future numerical comparison (currently unused).

    Returns
    -------
    bool
        True if verification passes.

    Raises
    ------
    AssertionError
        If the output shape does not match expectations.
    """
    import onnx
    import onnxruntime as ort

    onnx_path = Path(onnx_path)

    # Validate ONNX model structure
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    logger.info("ONNX model structure check passed")

    # Create dummy input
    if cfg.approach in ("pose_transformer", "pose_bilstm"):
        features_per_kp = 6 if getattr(cfg, "use_motion", False) else 3
        dummy = np.random.randn(1, cfg.T, cfg.num_keypoints * features_per_kp).astype(np.float32)
        input_name = "keypoints"
    elif cfg.approach == "video":
        dummy = np.random.randn(1, 3, cfg.T, cfg.image_size, cfg.image_size).astype(np.float32)
        input_name = "video"
    else:
        raise ValueError(f"Unsupported approach: {cfg.approach}")

    # Run ONNX Runtime
    session = ort.InferenceSession(str(onnx_path))
    ort_output = session.run(None, {input_name: dummy})[0]

    logger.info("ONNX output shape: %s", ort_output.shape)
    assert ort_output.shape == (1, cfg.num_classes), (
        f"Expected output shape (1, {cfg.num_classes}), got {ort_output.shape}"
    )

    logger.info("ONNX verification passed (output shape correct)")
    return True


def benchmark_onnx(
    onnx_path: str | Path,
    cfg: Config,
    n_runs: int = 100,
) -> dict[str, float]:
    """Benchmark inference latency of an ONNX model.

    Parameters
    ----------
    onnx_path : str or Path
        Path to the ONNX model.
    cfg : Config
        Configuration (for input shape).
    n_runs : int
        Number of inference runs to average.

    Returns
    -------
    dict[str, float]
        Keys: ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``, ``fps``.
    """
    import onnxruntime as ort

    onnx_path = Path(onnx_path)

    # Session options for optimal CPU performance
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts)

    # Create dummy input
    if cfg.approach in ("pose_transformer", "pose_bilstm"):
        features_per_kp = 6 if getattr(cfg, "use_motion", False) else 3
        dummy = np.random.randn(1, cfg.T, cfg.num_keypoints * features_per_kp).astype(np.float32)
        input_name = "keypoints"
    elif cfg.approach == "video":
        dummy = np.random.randn(1, 3, cfg.T, cfg.image_size, cfg.image_size).astype(np.float32)
        input_name = "video"
    else:
        raise ValueError(f"Unsupported approach: {cfg.approach}")

    # Warm-up
    for _ in range(10):
        session.run(None, {input_name: dummy})

    # Benchmark
    times_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_arr = np.array(times_ms)
    mean_ms = float(times_arr.mean())
    result = {
        "mean_ms": mean_ms,
        "std_ms": float(times_arr.std()),
        "min_ms": float(times_arr.min()),
        "max_ms": float(times_arr.max()),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }

    logger.info(
        "ONNX Latency: %.1f ms (std=%.1f) | FPS: %.1f",
        result["mean_ms"],
        result["std_ms"],
        result["fps"],
    )
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Verify after export")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after export")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    device = torch.device("cpu")

    # Build and load model
    if cfg.approach in ("pose_transformer", "pose_bilstm"):
        model = build_pose_model(cfg)
    elif cfg.approach == "video":
        model = build_video_model(cfg)
    else:
        raise ValueError(f"ONNX export not supported for approach '{cfg.approach}'")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Export
    onnx_path = export_to_onnx(model, cfg, args.output, opset=args.opset)

    # Verify
    if args.verify:
        verify_onnx(onnx_path, cfg)

    # Benchmark
    if args.benchmark:
        benchmark_onnx(onnx_path, cfg)
