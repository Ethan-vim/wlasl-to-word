"""
Evaluation utilities for trained WLASL recognition models.

Provides functions for computing classification metrics, plotting
confusion matrices, identifying hard negatives (confused class pairs),
and benchmarking inference latency.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augment import get_val_transforms
from src.data.dataset import (
    WLASLKeypointDataset,
    WLASLVideoDataset,
    WLASLFusionDataset,
    get_dataloader,
)
from src.models.pose_transformer import build_pose_model
from src.models.video_i3d import build_video_model
from src.models.fusion import build_fusion_model
from src.training.config import Config, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    approach: str = "pose_transformer",
) -> dict:
    """Compute comprehensive evaluation metrics.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.
    loader : DataLoader
        Evaluation DataLoader.
    device : torch.device
        Target device.
    class_names : list[str]
        List of class/gloss names, indexed by label.
    approach : str
        Model approach (``'fusion'`` requires dual-input unpacking).

    Returns
    -------
    dict
        Keys: ``top1``, ``top5``, ``per_class_accuracy``, ``confusion_matrix``,
        ``predictions``, ``targets``.
    """
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        if approach == "fusion":
            pose_input = batch[0].to(device, non_blocking=True)
            video_input = batch[1].to(device, non_blocking=True)
            targets = batch[2].to(device, non_blocking=True)
            logits = model(pose_input, video_input)
        else:
            inputs = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)
            logits = model(inputs)
        probs = torch.softmax(logits, dim=1)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        all_probs.append(probs.cpu().numpy())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    all_probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.array([])

    num_classes = len(class_names)
    num_samples = len(all_targets_arr)

    # Top-1 accuracy
    top1 = float(np.mean(all_preds_arr == all_targets_arr)) * 100.0 if num_samples > 0 else 0.0

    # Top-5 accuracy
    top5 = 0.0
    if num_samples > 0 and all_probs_arr.shape[0] > 0:
        top5_preds = np.argsort(all_probs_arr, axis=1)[:, -5:]
        top5_correct = np.array([
            all_targets_arr[i] in top5_preds[i] for i in range(num_samples)
        ])
        top5 = float(np.mean(top5_correct)) * 100.0

    # Per-class accuracy
    per_class_acc = {}
    for cls_idx in range(num_classes):
        mask = all_targets_arr == cls_idx
        if mask.sum() > 0:
            cls_acc = float(np.mean(all_preds_arr[mask] == cls_idx)) * 100.0
        else:
            cls_acc = 0.0
        name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        per_class_acc[name] = cls_acc

    # Confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(all_targets_arr, all_preds_arr, labels=list(range(num_classes)))

    return {
        "top1": top1,
        "top5": top5,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "predictions": all_preds_arr,
        "targets": all_targets_arr,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    figsize: Optional[tuple[int, int]] = None,
    max_classes_annotated: int = 30,
) -> None:
    """Plot and save a confusion matrix heatmap.

    For large numbers of classes, cell annotations are omitted to keep
    the plot readable.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape ``(num_classes, num_classes)``.
    class_names : list[str]
        Class names for axis labels.
    save_path : str or Path
        Path to save the figure.
    figsize : tuple[int, int] or None
        Figure size.  Defaults to a size proportional to the number of classes.
    max_classes_annotated : int
        Maximum number of classes for which to show cell value annotations.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_names)
    if figsize is None:
        side = max(8, min(n * 0.3, 40))
        figsize = (side, side)

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize by row (true class) for per-class recall
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums > 0,
    )

    annot = n <= max_classes_annotated
    sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt=".2f" if annot else "",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", save_path)


# ---------------------------------------------------------------------------
# Hard negatives
# ---------------------------------------------------------------------------


def find_hard_negatives(
    cm: np.ndarray,
    class_names: list[str],
    top_k: int = 10,
) -> list[tuple[str, str, int]]:
    """Identify the most commonly confused class pairs from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape ``(num_classes, num_classes)``.
    class_names : list[str]
        Class name list.
    top_k : int
        Number of top confused pairs to return.

    Returns
    -------
    list[tuple[str, str, int]]
        List of ``(true_class, predicted_class, count)`` sorted descending.
    """
    # Zero out the diagonal (correct predictions are not confusions)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

    # Find top-k off-diagonal entries
    flat_indices = np.argsort(cm_off.ravel())[::-1][:top_k]
    n = cm.shape[0]

    pairs = []
    for flat_idx in flat_indices:
        true_idx = flat_idx // n
        pred_idx = flat_idx % n
        count = int(cm_off[true_idx, pred_idx])
        if count == 0:
            break
        true_name = class_names[true_idx] if true_idx < len(class_names) else str(true_idx)
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        pairs.append((true_name, pred_name, count))

    return pairs


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------


def evaluate_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple[int, ...],
    n_runs: int = 100,
) -> dict[str, float]:
    """Measure inference latency of a model.

    Parameters
    ----------
    model : nn.Module
        Model to benchmark.
    device : torch.device
        Device to run on.
    input_shape : tuple[int, ...]
        Shape of a single input tensor (without batch dim).
    n_runs : int
        Number of forward passes to average over.

    Returns
    -------
    dict[str, float]
        Keys: ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``, ``fps``.
    """
    model.eval()
    dummy = torch.randn(1, *input_shape, device=device)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_arr = np.array(times_ms)
    mean_ms = float(times_arr.mean())
    return {
        "mean_ms": mean_ms,
        "std_ms": float(times_arr.std()),
        "min_ms": float(times_arr.min()),
        "max_ms": float(times_arr.max()),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_model(cfg: Config, device: torch.device) -> nn.Module:
    """Build and load a model from checkpoint."""
    if cfg.approach in ("pose_transformer", "pose_bilstm"):
        model = build_pose_model(cfg)
    elif cfg.approach == "video":
        model = build_video_model(cfg)
    elif cfg.approach == "fusion":
        pose_cfg = Config(
            approach="pose_transformer",
            num_keypoints=cfg.num_keypoints,
            num_classes=cfg.num_classes,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            T=cfg.T,
        )
        video_cfg = Config(
            approach="video",
            backbone=cfg.backbone,
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
            dropout=cfg.dropout,
        )
        pose_model = build_pose_model(pose_cfg)
        video_model = build_video_model(video_cfg)
        model = build_fusion_model(cfg, pose_model, video_model)
    else:
        raise ValueError(f"Unknown approach '{cfg.approach}'")
    return model.to(device)


def main() -> None:
    """CLI entry point for running evaluation on a saved checkpoint."""
    parser = argparse.ArgumentParser(description="Evaluate WLASL model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output-dir", type=str, default="eval_results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build and load model
    model = _build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build dataset
    data_dir = Path(cfg.data_dir)
    split_csv = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}" / f"{args.split}.csv"

    if cfg.approach in ("pose_transformer", "pose_bilstm"):
        transform = get_val_transforms(T=cfg.T)
        dataset = WLASLKeypointDataset(
            split_csv=split_csv,
            keypoint_dir=data_dir / "processed",
            transform=transform,
            T=cfg.T,
        )
    elif cfg.approach == "video":
        dataset = WLASLVideoDataset(
            split_csv=split_csv,
            video_dir=data_dir / "raw",
            T=cfg.T,
            size=cfg.image_size,
        )
    elif cfg.approach == "fusion":
        transform = get_val_transforms(T=cfg.T)
        dataset = WLASLFusionDataset(
            split_csv=split_csv,
            keypoint_dir=data_dir / "processed",
            video_dir=data_dir / "raw",
            kp_transform=transform,
            T_kp=cfg.T,
            T_vid=cfg.T // 2,
            size=cfg.image_size,
        )
    else:
        raise ValueError(f"Unknown approach '{cfg.approach}'")

    loader = get_dataloader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Build class names from all available split CSVs for complete coverage
    import pandas as pd
    class_names = [""] * cfg.num_classes
    splits_dir = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}"
    for split_name in ["train", "val", "test"]:
        csv_path = splits_dir / f"{split_name}.csv"
        if csv_path.exists():
            split_df = pd.read_csv(csv_path)
            for _, row in split_df.iterrows():
                idx = int(row["label_idx"])
                if idx < cfg.num_classes and not class_names[idx]:
                    class_names[idx] = row["gloss"]

    # Compute metrics (single inference pass)
    metrics = compute_metrics(model, loader, device, class_names, approach=cfg.approach)
    logger.info("Top-1 Accuracy: %.2f%%", metrics["top1"])
    logger.info("Top-5 Accuracy: %.2f%%", metrics["top5"])

    # Plot confusion matrix
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
    )

    # Hard negatives (reuses the already-computed confusion matrix — no re-inference)
    pairs = find_hard_negatives(metrics["confusion_matrix"], class_names, top_k=10)
    logger.info("Top confused pairs:")
    for true_cls, pred_cls, count in pairs:
        logger.info("  %s -> %s (%d times)", true_cls, pred_cls, count)

    # Latency benchmark (skipped for fusion — requires dual inputs)
    if cfg.approach in ("pose_transformer", "pose_bilstm"):
        input_shape = (cfg.T, cfg.num_keypoints * 3)
        latency = evaluate_latency(model, device, input_shape)
        logger.info(
            "Latency: %.1f ms (std=%.1f, min=%.1f, max=%.1f) | FPS: %.1f",
            latency["mean_ms"], latency["std_ms"],
            latency["min_ms"], latency["max_ms"],
            latency["fps"],
        )
    elif cfg.approach == "video":
        input_shape = (3, cfg.T, cfg.image_size, cfg.image_size)
        latency = evaluate_latency(model, device, input_shape)
        logger.info(
            "Latency: %.1f ms (std=%.1f, min=%.1f, max=%.1f) | FPS: %.1f",
            latency["mean_ms"], latency["std_ms"],
            latency["min_ms"], latency["max_ms"],
            latency["fps"],
        )
    else:
        logger.info("Latency benchmark skipped for fusion models (requires dual inputs)")


if __name__ == "__main__":
    main()
