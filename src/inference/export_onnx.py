"""
ONNX export, verification, and benchmarking for trained models.

Exports the ST-GCN encoder to ONNX format, verifies the exported model's
structure and output shape via ONNX Runtime, and benchmarks inference
latency.  Note: prototypes are stored separately (not part of the ONNX
graph) and must be loaded at inference time.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from src.models.prototypical import build_model
from src.training.config import Config, load_config

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    cfg: Config,
    output_path: str | Path,
    opset: int = 17,
) -> Path:
    """Export the ST-GCN encoder to ONNX format.

    The exported model takes keypoint sequences and produces embeddings.
    Prototype-based classification is done outside the ONNX graph.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = model.encoder if hasattr(model, "encoder") else model
    encoder.eval()
    encoder.cpu()

    features_per_kp = 6 if getattr(cfg, "use_motion", False) else 3
    dummy_input = torch.randn(1, cfg.T, cfg.num_keypoints * features_per_kp)
    input_names = ["keypoints"]
    dynamic_axes = {"keypoints": {0: "batch_size"}, "embedding": {0: "batch_size"}}

    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=input_names,
        output_names=["embedding"],
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
    """Verify an ONNX model's structure and output shape."""
    import onnx
    import onnxruntime as ort

    onnx_path = Path(onnx_path)

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    logger.info("ONNX model structure check passed")

    features_per_kp = 6 if getattr(cfg, "use_motion", False) else 3
    dummy = np.random.randn(1, cfg.T, cfg.num_keypoints * features_per_kp).astype(np.float32)

    session = ort.InferenceSession(str(onnx_path))
    ort_output = session.run(None, {"keypoints": dummy})[0]

    logger.info("ONNX output shape: %s", ort_output.shape)
    expected_dim = getattr(cfg, "embedding_dim", cfg.d_model)
    assert ort_output.shape == (1, expected_dim), (
        f"Expected output shape (1, {expected_dim}), got {ort_output.shape}"
    )

    logger.info("ONNX verification passed (output shape correct)")
    return True


def benchmark_onnx(
    onnx_path: str | Path,
    cfg: Config,
    n_runs: int = 100,
) -> dict[str, float]:
    """Benchmark inference latency of an ONNX model."""
    import onnxruntime as ort

    onnx_path = Path(onnx_path)

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts)

    features_per_kp = 6 if getattr(cfg, "use_motion", False) else 3
    dummy = np.random.randn(1, cfg.T, cfg.num_keypoints * features_per_kp).astype(np.float32)

    for _ in range(10):
        session.run(None, {"keypoints": dummy})

    times_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {"keypoints": dummy})
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
        result["mean_ms"], result["std_ms"], result["fps"],
    )
    return result


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

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    # Remove prototypes from checkpoint — they'll be recomputed from training data
    state_dict.pop("prototypes", None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    onnx_path = export_to_onnx(model, cfg, args.output, opset=args.opset)

    if args.verify:
        verify_onnx(onnx_path, cfg)

    if args.benchmark:
        benchmark_onnx(onnx_path, cfg)
