"""
PyTorch Dataset classes for WLASL keypoint and video data.

Provides ``WLASLKeypointDataset`` for Approach A (pose-based models) and
``WLASLVideoDataset`` for Approach B (RGB video models), along with a
factory function for creating DataLoaders with optional class-balanced
sampling.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class WLASLKeypointDataset(Dataset):
    """Dataset that loads precomputed ``.npy`` keypoint files.

    Each sample is a tuple ``(keypoints_tensor, label)`` where
    ``keypoints_tensor`` has shape ``(T, num_features)`` with features
    being the flattened ``(x, y, z)`` per landmark.

    Parameters
    ----------
    split_csv : str or Path
        Path to a CSV file with columns ``video_id``, ``label_idx``, ``gloss``.
    keypoint_dir : str or Path
        Directory containing ``{video_id}.npy`` files.
    transform : callable or None
        Augmentation pipeline (operates on shape ``(T_var, num_keypoints, 3)``).
    T : int
        Target sequence length.  If the transform does not include a
        ``TemporalCrop``, this dataset will pad/crop to ``T``.
    """

    def __init__(
        self,
        split_csv: str | Path,
        keypoint_dir: str | Path,
        transform: Optional[Callable] = None,
        T: int = 64,
    ) -> None:
        self.keypoint_dir = Path(keypoint_dir)
        self.transform = transform
        self.T = T

        df = pd.read_csv(split_csv)
        # Filter to only rows whose .npy file exists
        valid_mask = df["video_id"].apply(
            lambda vid: (self.keypoint_dir / f"{vid}.npy").exists()
        )
        self.df = df[valid_mask].reset_index(drop=True)
        if len(self.df) < len(df):
            logger.warning(
                "Filtered %d / %d samples (missing .npy files)",
                len(df) - len(self.df),
                len(df),
            )

        self.labels = self.df["label_idx"].values
        self.video_ids = self.df["video_id"].values
        self.glosses = self.df["gloss"].values

        # Build gloss-to-label mapping
        self.gloss_to_label: dict[str, int] = {}
        for _, row in self.df.iterrows():
            self.gloss_to_label[row["gloss"]] = int(row["label_idx"])

        self.num_classes = self.df["label_idx"].nunique()
        logger.info(
            "WLASLKeypointDataset: %d samples, %d classes", len(self.df), self.num_classes
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_id = self.video_ids[idx]
        label = int(self.labels[idx])

        npy_path = self.keypoint_dir / f"{video_id}.npy"
        keypoints = np.load(str(npy_path))  # (T_var, 543, 3)

        if self.transform is not None:
            keypoints = self.transform(keypoints)

        # If after transform the temporal length does not match T, fix it
        if keypoints.shape[0] != self.T:
            keypoints = self._pad_or_crop(keypoints)

        # Flatten spatial dims: (T, 543, 3) -> (T, 543*3)
        if keypoints.ndim == 3:
            T, K, C = keypoints.shape
            keypoints = keypoints.reshape(T, K * C)

        tensor = torch.from_numpy(keypoints).float()
        return tensor, label

    def _pad_or_crop(self, keypoints: np.ndarray) -> np.ndarray:
        """Ensure the sequence has exactly ``self.T`` frames."""
        T_in = keypoints.shape[0]
        if T_in == 0:
            return np.zeros((self.T, *keypoints.shape[1:]), dtype=np.float32)
        if T_in >= self.T:
            indices = np.linspace(0, T_in - 1, self.T, dtype=np.int64)
            return keypoints[indices]
        # Pad
        pad_count = self.T - T_in
        padding = np.tile(keypoints[-1:], (pad_count, *([1] * (keypoints.ndim - 1))))
        return np.concatenate([keypoints, padding], axis=0)


class WLASLVideoDataset(Dataset):
    """Dataset that loads raw video files and samples RGB frames.

    Each sample is a tuple ``(frames_tensor, label)`` where
    ``frames_tensor`` has shape ``(3, T, H, W)``.

    Parameters
    ----------
    split_csv : str or Path
        Path to a CSV with columns ``video_id``, ``label_idx``, ``gloss``.
    video_dir : str or Path
        Directory containing video files (``{video_id}.mp4`` etc.).
    transform : callable or None
        Optional spatial transform applied per-frame (e.g., albumentations).
    T : int
        Number of frames to sample per video.
    size : int
        Spatial resolution (frames are resized to ``size x size``).
    """

    def __init__(
        self,
        split_csv: str | Path,
        video_dir: str | Path,
        transform: Optional[Callable] = None,
        T: int = 32,
        size: int = 224,
    ) -> None:
        self.video_dir = Path(video_dir)
        self.transform = transform
        self.T = T
        self.size = size

        df = pd.read_csv(split_csv)
        # Filter to only rows with existing video files
        valid_rows = []
        self._video_paths: dict[str, Path] = {}
        for _, row in df.iterrows():
            vid = row["video_id"]
            vpath = self._find_video(vid)
            if vpath is not None:
                valid_rows.append(row)
                self._video_paths[vid] = vpath

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        if len(self.df) < len(df):
            logger.warning(
                "Filtered %d / %d samples (missing video files)",
                len(df) - len(self.df),
                len(df),
            )

        self.labels = self.df["label_idx"].values
        self.video_ids = self.df["video_id"].values
        self.num_classes = self.df["label_idx"].nunique()
        logger.info(
            "WLASLVideoDataset: %d samples, %d classes", len(self.df), self.num_classes
        )

    def _find_video(self, video_id: str) -> Optional[Path]:
        """Search for a video file across common extensions."""
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            p = self.video_dir / f"{video_id}{ext}"
            if p.exists():
                return p
        return None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_id = self.video_ids[idx]
        label = int(self.labels[idx])
        video_path = self._video_paths[video_id]

        frames = self._load_frames(video_path)
        # frames shape: (T, H, W, 3) uint8

        if self.transform is not None:
            # Apply spatial transform per frame
            transformed = []
            for i in range(frames.shape[0]):
                result = self.transform(image=frames[i])
                transformed.append(result["image"])
            frames = np.stack(transformed, axis=0)

        # Normalize to [0, 1] and convert to (3, T, H, W)
        tensor = torch.from_numpy(frames).float() / 255.0
        tensor = tensor.permute(3, 0, 1, 2)  # (T, H, W, 3) -> (3, T, H, W)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, label

    def _load_frames(self, video_path: Path) -> np.ndarray:
        """Load and uniformly sample T frames from a video.

        Attempts to use Decord for fast video loading; falls back to
        OpenCV if Decord is unavailable.

        Returns
        -------
        np.ndarray
            Shape ``(T, size, size, 3)`` in uint8 RGB format.
        """
        try:
            return self._load_frames_decord(video_path)
        except (ImportError, RuntimeError):
            return self._load_frames_opencv(video_path)

    def _load_frames_decord(self, video_path: Path) -> np.ndarray:
        """Load frames using Decord for fast random-access decoding."""
        import decord
        decord.bridge.set_bridge("numpy")

        vr = decord.VideoReader(str(video_path), width=self.size, height=self.size)
        total = len(vr)
        if total == 0:
            return np.zeros((self.T, self.size, self.size, 3), dtype=np.uint8)

        indices = np.linspace(0, total - 1, self.T, dtype=np.int64)
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) RGB — keep as RGB
        return frames

    def _load_frames_opencv(self, video_path: Path) -> np.ndarray:
        """Fallback frame loader using OpenCV."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        all_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV reads BGR; convert to RGB for ImageNet normalization
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.size, self.size))
            all_frames.append(frame)
        cap.release()

        if len(all_frames) == 0:
            return np.zeros((self.T, self.size, self.size, 3), dtype=np.uint8)

        all_frames = np.stack(all_frames, axis=0)
        total = all_frames.shape[0]
        indices = np.linspace(0, total - 1, self.T, dtype=np.int64)
        return all_frames[indices]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


class WLASLFusionDataset(Dataset):
    """Dataset that loads both keypoints and RGB video frames for fusion models.

    Each sample is a tuple ``(keypoints_tensor, video_tensor, label)`` where
    ``keypoints_tensor`` is shape ``(T_kp, num_features)`` and
    ``video_tensor`` is shape ``(3, T_vid, H, W)``.

    Parameters
    ----------
    split_csv : str or Path
        CSV with columns ``video_id``, ``label_idx``, ``gloss``.
    keypoint_dir : str or Path
        Directory containing ``{video_id}.npy`` keypoint files.
    video_dir : str or Path
        Directory containing raw video files.
    kp_transform : callable or None
        Augmentation pipeline for keypoints.
    T_kp : int
        Keypoint sequence length.
    T_vid : int
        Number of video frames to sample.
    size : int
        Spatial resolution for video frames.
    """

    def __init__(
        self,
        split_csv: str | Path,
        keypoint_dir: str | Path,
        video_dir: str | Path,
        kp_transform: Optional[Callable] = None,
        T_kp: int = 64,
        T_vid: int = 32,
        size: int = 224,
    ) -> None:
        self.keypoint_dir = Path(keypoint_dir)
        self.video_dir = Path(video_dir)
        self.kp_transform = kp_transform
        self.T_kp = T_kp
        self.T_vid = T_vid
        self.size = size

        df = pd.read_csv(split_csv)
        # Keep only rows where both .npy and video file exist
        valid_rows = []
        self._video_paths: dict[str, Path] = {}
        for _, row in df.iterrows():
            vid = row["video_id"]
            npy = self.keypoint_dir / f"{vid}.npy"
            vpath = self._find_video(vid)
            if npy.exists() and vpath is not None:
                valid_rows.append(row)
                self._video_paths[vid] = vpath

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        if len(self.df) < len(df):
            logger.warning(
                "WLASLFusionDataset: filtered %d / %d samples (missing .npy or video)",
                len(df) - len(self.df),
                len(df),
            )

        self.labels = self.df["label_idx"].values
        self.video_ids = self.df["video_id"].values
        self.num_classes = self.df["label_idx"].nunique()
        logger.info(
            "WLASLFusionDataset: %d samples, %d classes", len(self.df), self.num_classes
        )

    def _find_video(self, video_id: str) -> Optional[Path]:
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            p = self.video_dir / f"{video_id}{ext}"
            if p.exists():
                return p
        return None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        video_id = self.video_ids[idx]
        label = int(self.labels[idx])

        # Load keypoints
        keypoints = np.load(str(self.keypoint_dir / f"{video_id}.npy"))
        if self.kp_transform is not None:
            keypoints = self.kp_transform(keypoints)
        if keypoints.shape[0] != self.T_kp:
            keypoints = _pad_or_crop_seq(keypoints, self.T_kp)
        if keypoints.ndim == 3:
            T, K, C = keypoints.shape
            keypoints = keypoints.reshape(T, K * C)
        kp_tensor = torch.from_numpy(keypoints).float()

        # Load video frames
        vid_tensor = self._load_video_tensor(self._video_paths[video_id])

        return kp_tensor, vid_tensor, label

    def _load_video_tensor(self, video_path: Path) -> torch.Tensor:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.size, self.size))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return torch.zeros(3, self.T_vid, self.size, self.size)

        frames_arr = np.stack(frames, axis=0)
        indices = np.linspace(0, len(frames_arr) - 1, self.T_vid, dtype=np.int64)
        frames_arr = frames_arr[indices]

        tensor = torch.from_numpy(frames_arr).float() / 255.0
        tensor = tensor.permute(3, 0, 1, 2)  # (3, T, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        return (tensor - mean) / std


def _pad_or_crop_seq(keypoints: np.ndarray, T: int) -> np.ndarray:
    """Pad or uniformly crop a keypoint sequence to exactly T frames."""
    T_in = keypoints.shape[0]
    if T_in == 0:
        return np.zeros((T, *keypoints.shape[1:]), dtype=np.float32)
    if T_in >= T:
        indices = np.linspace(0, T_in - 1, T, dtype=np.int64)
        return keypoints[indices]
    pad_count = T - T_in
    padding = np.tile(keypoints[-1:], (pad_count, *([1] * (keypoints.ndim - 1))))
    return np.concatenate([keypoints, padding], axis=0)


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    weighted_sampling: bool = False,
) -> DataLoader:
    """Create a DataLoader with optional class-balanced weighted sampling.

    When ``weighted_sampling`` is True, each sample's weight is the
    inverse of its class frequency, ensuring balanced mini-batches.
    ``shuffle`` is automatically disabled when a sampler is used.

    Parameters
    ----------
    dataset : Dataset
        A WLASL dataset instance.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Whether to shuffle (ignored if ``weighted_sampling`` is True).
    num_workers : int
        Number of data loading workers.
    weighted_sampling : bool
        If True, use ``WeightedRandomSampler`` for class imbalance.

    Returns
    -------
    DataLoader
    """
    sampler = None
    if weighted_sampling and hasattr(dataset, "labels"):
        labels = dataset.labels
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(dataset),
            replacement=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
        persistent_workers=num_workers > 0,
    )
