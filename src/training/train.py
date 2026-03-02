"""
Full training loop for WLASL sign language recognition.

Supports all three approaches (pose, video, fusion) with mixed-precision
training, cosine-annealing + warm-up scheduling, gradient clipping,
checkpoint saving, early stopping, and optional W&B / TensorBoard logging.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augment import get_train_transforms, get_val_transforms
from src.data.dataset import (
    WLASLKeypointDataset,
    WLASLVideoDataset,
    WLASLFusionDataset,
    get_dataloader,
)
from src.models.fusion import build_fusion_model
from src.models.pose_transformer import build_pose_model
from src.models.video_i3d import build_video_model
from src.training.config import Config, load_config, save_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1, 5)) -> list[float]:
    """Compute top-k accuracy for the given output and target tensors.

    Parameters
    ----------
    output : torch.Tensor
        Model logits of shape ``(B, num_classes)``.
    target : torch.Tensor
        Ground-truth labels of shape ``(B,)``.
    topk : tuple of int
        Which top-k accuracies to compute.

    Returns
    -------
    list of float
        Accuracy values for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if batch_size == 0:
            return [0.0] * len(topk)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.unsqueeze(0).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------


def _mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup to a batch: linearly interpolate random pairs.

    Parameters
    ----------
    x : torch.Tensor
        Input batch ``(B, ...)``.
    y : torch.Tensor
        Label batch ``(B,)`` (integer class indices).
    alpha : float
        Beta distribution parameter.  Higher values produce more mixing.

    Returns
    -------
    tuple
        ``(mixed_x, y_a, y_b, lam)`` where ``mixed_x`` is the interpolated
        input, ``y_a`` and ``y_b`` are the original and shuffled labels,
        and ``lam`` is the mixing coefficient.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def _mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute loss for mixed labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Training and validation steps
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    epoch: int = 0,
    writer: Optional[object] = None,
    global_step: int = 0,
) -> tuple[float, float, float, int]:
    """Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    loader : DataLoader
        Training DataLoader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : LRScheduler or None
        Learning rate scheduler (stepped per batch for OneCycleLR).
    scaler : GradScaler or None
        AMP gradient scaler (None if FP16 is disabled).
    criterion : nn.Module
        Loss function.
    device : torch.device
        Target device.
    cfg : Config
        Configuration.
    epoch : int
        Current epoch number (for logging).
    writer : SummaryWriter or None
        TensorBoard writer.
    global_step : int
        Current global step count.

    Returns
    -------
    tuple[float, float, float, int]
        (avg_loss, top1_acc, top5_acc, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for batch in pbar:
        if cfg.approach == "fusion":
            # WLASLFusionDataset returns (kp, video, label)
            pose_input = batch[0].to(device, non_blocking=True)
            video_input = batch[1].to(device, non_blocking=True)
            targets = batch[2].to(device, non_blocking=True)
        else:
            pose_input = batch[0].to(device, non_blocking=True)
            video_input = None
            targets = batch[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixup
        use_mixup = cfg.mixup_alpha > 0
        if use_mixup:
            pose_input, targets_a, targets_b, lam = _mixup_data(
                pose_input, targets, alpha=cfg.mixup_alpha
            )

        use_amp = cfg.fp16 and device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            if cfg.approach == "fusion":
                logits = model(pose_input, video_input)
            else:
                logits = model(pose_input)
            if use_mixup:
                loss = _mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        top1, top5 = _accuracy(logits, targets, topk=(1, 5))
        total_loss += loss.item()
        total_top1 += top1
        total_top5 += top5
        num_batches += 1
        global_step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", top1=f"{top1:.1f}%")

        # TensorBoard step-level logging
        if writer is not None and global_step % cfg.log_interval == 0:
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/top1_step", top1, global_step)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", current_lr, global_step)

    avg_loss = total_loss / max(num_batches, 1)
    avg_top1 = total_top1 / max(num_batches, 1)
    avg_top5 = total_top5 / max(num_batches, 1)

    return avg_loss, avg_top1, avg_top5, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
) -> tuple[float, float, float, np.ndarray]:
    """Run validation and return metrics + confusion matrix.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        Validation DataLoader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Target device.
    cfg : Config
        Configuration.

    Returns
    -------
    tuple[float, float, float, np.ndarray]
        (avg_loss, top1_acc, top5_acc, confusion_matrix)
    """
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        if cfg.approach == "fusion":
            pose_input = batch[0].to(device, non_blocking=True)
            video_input = batch[1].to(device, non_blocking=True)
            targets = batch[2].to(device, non_blocking=True)
            logits = model(pose_input, video_input)
        else:
            logits = model(batch[0].to(device, non_blocking=True))
            targets = batch[1].to(device, non_blocking=True)

        loss = criterion(logits, targets)

        top1, top5 = _accuracy(logits, targets, topk=(1, 5))
        total_loss += loss.item()
        total_top1 += top1
        total_top5 += top5
        num_batches += 1

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    avg_top1 = total_top1 / max(num_batches, 1)
    avg_top5 = total_top5 / max(num_batches, 1)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix as _cm

    if len(all_preds) > 0:
        cm = _cm(all_targets, all_preds, labels=list(range(cfg.num_classes)))
    else:
        cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)

    return avg_loss, avg_top1, avg_top5, cm


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_model(cfg: Config, device: torch.device) -> nn.Module:
    """Instantiate the model according to the configured approach."""
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


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _build_datasets(cfg: Config) -> tuple:
    """Create train and val datasets from config."""
    data_dir = Path(cfg.data_dir)
    splits_dir = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}"
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    # Check that split CSVs exist before trying to build datasets
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Training split not found: {train_csv}\n"
            f"Run preprocessing first:\n"
            f"  python -m src.data.preprocess --data-dir {cfg.data_dir} "
            f"--subset WLASL{cfg.wlasl_variant}"
        )
    if not val_csv.exists():
        raise FileNotFoundError(
            f"Validation split not found: {val_csv}\n"
            f"Run preprocessing first:\n"
            f"  python -m src.data.preprocess --data-dir {cfg.data_dir} "
            f"--subset WLASL{cfg.wlasl_variant}"
        )

    if cfg.approach in ("pose_transformer", "pose_bilstm"):
        train_transform = get_train_transforms(T=cfg.T)
        val_transform = get_val_transforms(T=cfg.T)

        train_ds = WLASLKeypointDataset(
            split_csv=train_csv,
            keypoint_dir=processed_dir,
            transform=train_transform,
            T=cfg.T,
            use_motion=cfg.use_motion,
        )
        val_ds = WLASLKeypointDataset(
            split_csv=val_csv,
            keypoint_dir=processed_dir,
            transform=val_transform,
            T=cfg.T,
            use_motion=cfg.use_motion,
        )
    elif cfg.approach == "video":
        train_ds = WLASLVideoDataset(
            split_csv=train_csv,
            video_dir=raw_dir,
            T=cfg.T,
            size=cfg.image_size,
        )
        val_ds = WLASLVideoDataset(
            split_csv=val_csv,
            video_dir=raw_dir,
            T=cfg.T,
            size=cfg.image_size,
        )
    elif cfg.approach == "fusion":
        train_transform = get_train_transforms(T=cfg.T)
        val_transform = get_val_transforms(T=cfg.T)
        train_ds = WLASLFusionDataset(
            split_csv=train_csv,
            keypoint_dir=processed_dir,
            video_dir=raw_dir,
            kp_transform=train_transform,
            T_kp=cfg.T,
            T_vid=cfg.T // 2,
            size=cfg.image_size,
        )
        val_ds = WLASLFusionDataset(
            split_csv=val_csv,
            keypoint_dir=processed_dir,
            video_dir=raw_dir,
            kp_transform=val_transform,
            T_kp=cfg.T,
            T_vid=cfg.T // 2,
            size=cfg.image_size,
        )
    else:
        raise ValueError(f"Unknown approach '{cfg.approach}'")

    # Log data coverage diagnostics
    train_classes = getattr(train_ds, "num_classes", None)
    train_len = len(train_ds)
    val_len = len(val_ds)
    if train_classes is not None and train_classes < cfg.num_classes:
        logger.warning(
            "Only %d / %d classes have training data. Classes without samples "
            "will not be learned. Consider downloading more videos or using a "
            "smaller wlasl_variant.",
            train_classes,
            cfg.num_classes,
        )
    if train_len > 0 and train_len < cfg.num_classes * 3:
        logger.warning(
            "Very few training samples (%d) for %d classes (%.1f per class). "
            "Expect limited accuracy. See README 'Training with Limited Data' "
            "for tips.",
            train_len,
            cfg.num_classes,
            train_len / cfg.num_classes,
        )
    logger.info(
        "Dataset: %d train / %d val samples",
        train_len,
        val_len,
    )

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def main(cfg: Config) -> None:
    """Run the full training pipeline.

    Parameters
    ----------
    cfg : Config
        Fully populated configuration.
    """
    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(cfg, checkpoint_dir / "config.yaml")

    # TensorBoard
    writer = None
    if cfg.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))

    # W&B
    if cfg.use_wandb:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=vars(cfg) if hasattr(cfg, "__dict__") else {},
            )
        except ImportError:
            logger.warning("wandb not installed; disabling W&B logging")
            cfg.use_wandb = False

    # Data
    train_ds, val_ds = _build_datasets(cfg)
    train_loader = get_dataloader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        weighted_sampling=cfg.weighted_sampling,
    )
    val_loader = get_dataloader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # Model
    model = _build_model(cfg, device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Guard against an empty training set (e.g., all videos missing)
    if len(train_loader) == 0:
        logger.error(
            "Training DataLoader is empty. Check that split CSVs exist at '%s' "
            "and that processed keypoints/videos are present.",
            Path(cfg.data_dir) / "splits" / f"WLASL{cfg.wlasl_variant}",
        )
        return

    # Scheduler
    total_steps = cfg.epochs * len(train_loader)
    if cfg.scheduler == "cosine":
        warmup_steps = cfg.warmup_epochs * len(train_loader)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:  # onecycle (default)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=cfg.warmup_epochs / max(cfg.epochs, 1),
            anneal_strategy="cos",
        )

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # AMP scaler
    scaler = GradScaler() if cfg.fp16 and device.type == "cuda" else None

    # Resume from checkpoint
    start_epoch = 0
    best_top1 = 0.0
    global_step = 0
    if cfg.resume_checkpoint is not None:
        ckpt_path = Path(cfg.resume_checkpoint)
        if ckpt_path.exists():
            logger.info("Resuming from checkpoint: %s", ckpt_path)
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt and scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_top1 = ckpt.get("best_top1", 0.0)
            global_step = ckpt.get("global_step", 0)
        else:
            logger.warning("Checkpoint not found at %s, training from scratch", ckpt_path)

    # Training loop
    epochs_without_improvement = 0
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        train_loss, train_top1, train_top5, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            device, cfg, epoch=epoch, writer=writer, global_step=global_step,
        )

        val_loss, val_top1, val_top5, cm = validate(
            model, val_loader, criterion, device, cfg,
        )

        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d (%.1fs) | "
            "Train Loss: %.4f Top1: %.1f%% Top5: %.1f%% | "
            "Val Loss: %.4f Top1: %.1f%% Top5: %.1f%%",
            epoch + 1, cfg.epochs, elapsed,
            train_loss, train_top1, train_top5,
            val_loss, val_top1, val_top5,
        )

        # TensorBoard epoch-level logging
        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("train/top1_epoch", train_top1, epoch)
            writer.add_scalar("train/top5_epoch", train_top5, epoch)
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            writer.add_scalar("val/top1_epoch", val_top1, epoch)
            writer.add_scalar("val/top5_epoch", val_top5, epoch)

        # W&B logging
        if cfg.use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/top1": train_top1,
                    "train/top5": train_top5,
                    "val/loss": val_loss,
                    "val/top1": val_top1,
                    "val/top5": val_top5,
                },
                step=global_step,
            )

        # Save best checkpoint
        is_best = val_top1 > best_top1
        if is_best:
            best_top1 = val_top1
            epochs_without_improvement = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_top1": best_top1,
                "global_step": global_step,
                "config": vars(cfg) if hasattr(cfg, "__dict__") else {},
            }
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(ckpt, str(best_path))
            logger.info("Saved best model (Top1: %.1f%%) to %s", best_top1, best_path)
        else:
            epochs_without_improvement += 1

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_top1": best_top1,
                "global_step": global_step,
            }
            periodic_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(ckpt, str(periodic_path))

        # Early stopping
        if cfg.early_stopping_patience > 0 and epochs_without_improvement >= cfg.early_stopping_patience:
            logger.info(
                "Early stopping after %d epochs without improvement",
                cfg.early_stopping_patience,
            )
            break

    logger.info("Training complete. Best validation Top-1: %.1f%%", best_top1)

    if writer is not None:
        writer.close()
    if cfg.use_wandb:
        import wandb

        wandb.finish()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WLASL recognition model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    main(cfg)
