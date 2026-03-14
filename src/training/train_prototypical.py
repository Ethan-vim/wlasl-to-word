"""
Episodic training loop for Prototypical Network + ST-GCN.

Uses N-way K-shot episodes for training, where each episode samples
N classes with K support + Q query samples.  Validation computes
prototypes from the full training set and classifies via nearest-
prototype (Euclidean distance).
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.data.augment import get_train_transforms, get_val_transforms
from src.data.dataset import WLASLKeypointDataset, get_dataloader
from src.data.episode_sampler import EpisodicBatchSampler
from src.models.prototypical import PrototypicalNetwork, build_model
from src.training.config import Config, load_config, save_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------


def _split_episode(
    x: torch.Tensor,
    y: torch.Tensor,
    n_way: int,
    k_shot: int,
    q_query: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a flat episode batch into support / query with local labels.

    The batch is assumed to be ordered as N_way groups of (K_shot + Q_query)
    samples each, as produced by ``EpisodicBatchSampler``.

    Returns
    -------
    support_x, support_y, query_x, query_y
        support_y and query_y use episode-local labels 0..N_way-1.
    """
    per_class = k_shot + q_query
    support_x_parts: list[torch.Tensor] = []
    query_x_parts: list[torch.Tensor] = []
    support_y_list: list[int] = []
    query_y_list: list[int] = []

    for i in range(n_way):
        start = i * per_class
        support_x_parts.append(x[start : start + k_shot])
        query_x_parts.append(x[start + k_shot : start + per_class])
        support_y_list.extend([i] * k_shot)
        query_y_list.extend([i] * q_query)

    support_x = torch.cat(support_x_parts, dim=0)
    query_x = torch.cat(query_x_parts, dim=0)
    support_y = torch.tensor(support_y_list, dtype=torch.long, device=x.device)
    query_y = torch.tensor(query_y_list, dtype=torch.long, device=x.device)

    return support_x, support_y, query_x, query_y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: PrototypicalNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    epoch: int = 0,
    writer: Optional[object] = None,
    global_step: int = 0,
) -> tuple[float, float, int]:
    """Train for one epoch of episodes.

    Returns
    -------
    tuple[float, float, int]
        (avg_loss, avg_query_accuracy, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_episodes = 0

    non_blocking = device.type == "cuda"
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False, dynamic_ncols=True)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device, non_blocking=non_blocking)

        support_x, support_y, query_x, query_y = _split_episode(
            batch_x, batch_y, cfg.n_way, cfg.k_shot, cfg.q_query
        )

        optimizer.zero_grad(set_to_none=True)

        # Forward: returns negative distances (N*Q, N_way)
        logits = model(support_x, support_y, query_x)
        loss = F.cross_entropy(logits, query_y)

        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # MPS queues Metal commands asynchronously — flush every step to
        # prevent the command queue from filling up and stalling.
        if device.type == "mps":
            torch.mps.synchronize()

        # Episode accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == query_y).float().mean().item() * 100.0

        total_loss += loss.item()
        total_acc += acc
        num_episodes += 1
        global_step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.1f}%")

        if writer is not None and global_step % cfg.log_interval == 0:
            writer.add_scalar("train/episode_loss", loss.item(), global_step)
            writer.add_scalar("train/episode_acc", acc, global_step)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", current_lr, global_step)

    avg_loss = total_loss / max(num_episodes, 1)
    avg_acc = total_acc / max(num_episodes, 1)
    return avg_loss, avg_acc, global_step


# ---------------------------------------------------------------------------
# Validation (prototype-based)
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: PrototypicalNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> tuple[float, float]:
    """Validate by computing prototypes from train set and classifying val.

    Returns
    -------
    tuple[float, float]
        (top1_accuracy, top5_accuracy)
    """
    model.eval()

    # Compute prototypes from full training set
    model.compute_prototypes(train_loader)

    all_preds: list[int] = []
    all_targets: list[int] = []
    all_logits: list[torch.Tensor] = []

    non_blocking = device.type == "cuda"
    for batch_x, batch_y in tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=non_blocking)
        logits = model.classify(batch_x)  # (B, num_classes)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(batch_y.tolist())
        all_logits.append(logits.cpu())

        if device.type == "mps":
            torch.mps.synchronize()

    if not all_targets:
        return 0.0, 0.0

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    all_logits_cat = torch.cat(all_logits, dim=0)

    # Top-1
    top1 = float(np.mean(all_preds_arr == all_targets_arr)) * 100.0

    # Top-5
    top5_preds = all_logits_cat.topk(min(5, all_logits_cat.size(1)), dim=1).indices.numpy()
    top5_correct = np.array([
        all_targets_arr[i] in top5_preds[i] for i in range(len(all_targets_arr))
    ])
    top5 = float(np.mean(top5_correct)) * 100.0

    return top1, top5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config) -> None:
    """Run the full prototypical training pipeline."""
    # Device
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

    # Datasets
    data_dir = Path(cfg.data_dir)
    splits_dir = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}"
    processed_dir = data_dir / "processed"

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Training split not found: {train_csv}\n"
            f"Run preprocessing first:\n"
            f"  python -m src.data.preprocess --data-dir {cfg.data_dir} "
            f"--subset WLASL{cfg.wlasl_variant}"
        )

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

    logger.info("Dataset: %d train / %d val samples", len(train_ds), len(val_ds))

    # Episodic training loader
    episode_sampler = EpisodicBatchSampler(
        labels=train_ds.labels,
        n_way=cfg.n_way,
        k_shot=cfg.k_shot,
        q_query=cfg.q_query,
        num_episodes=cfg.num_episodes,
    )
    use_pin_memory = device.type == "cuda"
    # MPS backend deadlocks with multiprocessing workers — force num_workers=0
    num_workers = cfg.num_workers if device.type != "mps" else 0
    train_loader = DataLoader(
        train_ds,
        batch_sampler=episode_sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    # Standard loader for prototype computation + validation
    proto_loader = get_dataloader(
        train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers,
    )
    val_loader = get_dataloader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers,
    )

    # Model
    model = build_model(cfg).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # Scheduler (cosine annealing over total episodes)
    total_episodes = cfg.epochs * cfg.num_episodes
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_episodes, eta_min=cfg.lr * 0.01,
    )

    # Resume
    start_epoch = 0
    best_top1 = 0.0
    global_step = 0
    if cfg.resume_checkpoint is not None:
        ckpt_path = Path(cfg.resume_checkpoint)
        if ckpt_path.exists():
            logger.info("Resuming from %s", ckpt_path)
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
    with logging_redirect_tqdm():
        for epoch in range(start_epoch, cfg.epochs):
            t0 = time.time()

            train_loss, train_acc, global_step = train_one_epoch(
                model, train_loader, optimizer, device, cfg,
                epoch=epoch, writer=writer, global_step=global_step,
            )

            # Step scheduler per epoch (covers all episodes in that epoch)
            scheduler.step()

            val_top1, val_top5 = validate(
                model, proto_loader, val_loader, device, cfg,
            )

            elapsed = time.time() - t0

            logger.info(
                "Epoch %d/%d (%.1fs) | "
                "Train Loss: %.4f Acc: %.1f%% | "
                "Val Top1: %.1f%% Top5: %.1f%%",
                epoch + 1, cfg.epochs, elapsed,
                train_loss, train_acc,
                val_top1, val_top5,
            )

            # TensorBoard
            if writer is not None:
                writer.add_scalar("train/loss_epoch", train_loss, epoch)
                writer.add_scalar("train/acc_epoch", train_acc, epoch)
                writer.add_scalar("val/top1_epoch", val_top1, epoch)
                writer.add_scalar("val/top5_epoch", val_top5, epoch)

            # W&B
            if cfg.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/top1": val_top1,
                    "val/top5": val_top5,
                }, step=global_step)

            # Save best
            is_best = val_top1 > best_top1
            if is_best:
                best_top1 = val_top1
                epochs_without_improvement = 0
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_top1": best_top1,
                    "global_step": global_step,
                    "config": vars(cfg) if hasattr(cfg, "__dict__") else {},
                }
                best_path = checkpoint_dir / "best_model.pt"
                torch.save(ckpt, str(best_path))
                logger.info("Saved best model (Top1: %.1f%%) to %s", best_top1, best_path)
            else:
                epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_top1": best_top1,
                    "global_step": global_step,
                }
                torch.save(ckpt, str(checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"))

            # Early stopping
            if (
                cfg.early_stopping_patience > 0
                and epochs_without_improvement >= cfg.early_stopping_patience
            ):
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
