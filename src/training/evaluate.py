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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augment import get_val_transforms, KeypointHorizontalFlip
from src.data.dataset import WLASLKeypointDataset, get_dataloader
from src.models.prototypical import PrototypicalNetwork, build_model
from src.training.config import Config, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

_hflip = KeypointHorizontalFlip(p=1.0, centered=True)


def _flip_keypoints_tensor(x: torch.Tensor, num_keypoints: int = 543) -> torch.Tensor:
    """Horizontally flip a batch of flattened keypoint tensors."""
    B, T, F = x.shape
    C = F // num_keypoints
    flipped = x.clone().cpu().numpy()
    for i in range(B):
        sample = flipped[i]
        if C == 6:
            sample_3d = sample.reshape(T, num_keypoints, C)
            pos = sample_3d[:, :, :3].copy()
            vel = sample_3d[:, :, 3:].copy()
            pos = _hflip(pos)
            vel = _hflip(vel)
            sample_3d = np.concatenate([pos, vel], axis=-1)
            flipped[i] = sample_3d.reshape(T, -1)
        else:
            sample_3d = sample.reshape(T, num_keypoints, 3)
            sample_3d = _hflip(sample_3d)
            flipped[i] = sample_3d.reshape(T, -1)
    return torch.from_numpy(flipped).to(x.device)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_metrics(
    model: PrototypicalNetwork,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    use_tta: bool = False,
    num_keypoints: int = 543,
) -> dict:
    """Compute comprehensive evaluation metrics using prototype classification."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        inputs = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        logits = model.classify(inputs)

        if use_tta:
            flipped = _flip_keypoints_tensor(inputs, num_keypoints=num_keypoints)
            logits_flip = model.classify(flipped)
            logits = (logits + logits_flip) / 2.0

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

    top1 = float(np.mean(all_preds_arr == all_targets_arr)) * 100.0 if num_samples > 0 else 0.0

    top5 = 0.0
    if num_samples > 0 and all_probs_arr.shape[0] > 0:
        top5_preds = np.argsort(all_probs_arr, axis=1)[:, -5:]
        top5_correct = np.array([
            all_targets_arr[i] in top5_preds[i] for i in range(num_samples)
        ])
        top5 = float(np.mean(top5_correct)) * 100.0

    per_class_acc = {}
    for cls_idx in range(num_classes):
        mask = all_targets_arr == cls_idx
        if mask.sum() > 0:
            cls_acc = float(np.mean(all_preds_arr[mask] == cls_idx)) * 100.0
        else:
            cls_acc = 0.0
        name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        per_class_acc[name] = cls_acc

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
    """Plot and save a confusion matrix heatmap."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_names)
    if figsize is None:
        side = max(8, min(n * 0.3, 40))
        figsize = (side, side)

    fig, ax = plt.subplots(figsize=figsize)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(np.float64), row_sums,
        out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0,
    )

    annot = n <= max_classes_annotated
    sns.heatmap(
        cm_normalized, annot=annot, fmt=".2f" if annot else "",
        cmap="Blues", xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0.0, vmax=1.0,
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
    """Identify the most commonly confused class pairs."""
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

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
    """Measure inference latency of the encoder."""
    model.eval()
    dummy = torch.randn(1, *input_shape, device=device)

    for _ in range(10):
        with torch.no_grad():
            _ = model.classify(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    times_ms: list[float] = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.classify(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Build and load model
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build dataset
    data_dir = Path(cfg.data_dir)
    split_csv = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}" / f"{args.split}.csv"
    train_csv = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}" / "train.csv"

    transform = get_val_transforms(T=cfg.T)
    dataset = WLASLKeypointDataset(
        split_csv=split_csv,
        keypoint_dir=data_dir / "processed",
        transform=transform,
        T=cfg.T,
        use_motion=getattr(cfg, "use_motion", False),
    )
    train_ds = WLASLKeypointDataset(
        split_csv=train_csv,
        keypoint_dir=data_dir / "processed",
        transform=transform,
        T=cfg.T,
        use_motion=getattr(cfg, "use_motion", False),
    )

    loader = get_dataloader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    proto_loader = get_dataloader(train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Compute prototypes from training data
    model.compute_prototypes(proto_loader)

    # Build class names
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

    # Compute metrics
    metrics = compute_metrics(
        model, loader, device, class_names,
        use_tta=getattr(cfg, "use_tta", False),
        num_keypoints=cfg.num_keypoints,
    )
    logger.info("Top-1 Accuracy: %.2f%%", metrics["top1"])
    logger.info("Top-5 Accuracy: %.2f%%", metrics["top5"])

    # Confusion matrix
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, output_dir / "confusion_matrix.png")

    # Hard negatives
    pairs = find_hard_negatives(metrics["confusion_matrix"], class_names, top_k=10)
    logger.info("Top confused pairs:")
    for true_cls, pred_cls, count in pairs:
        logger.info("  %s -> %s (%d times)", true_cls, pred_cls, count)

    # Latency benchmark
    features_per_kp = 6 if getattr(cfg, "use_motion", False) else 3
    input_shape = (cfg.T, cfg.num_keypoints * features_per_kp)
    latency = evaluate_latency(model, device, input_shape)
    logger.info(
        "Latency: %.1f ms (std=%.1f, min=%.1f, max=%.1f) | FPS: %.1f",
        latency["mean_ms"], latency["std_ms"],
        latency["min_ms"], latency["max_ms"],
        latency["fps"],
    )


if __name__ == "__main__":
    main()
