"""
Single-video prediction module.

Provides ``SignPredictor`` for running inference on individual video files
or precomputed keypoint arrays, returning the predicted gloss with
confidence scores and top-5 alternatives.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.data.augment import get_val_transforms
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.data.preprocess import (
    extract_keypoints_mediapipe,
    normalize_keypoints,
    NUM_KEYPOINTS,
)
from src.models.pose_transformer import build_pose_model
from src.models.video_i3d import build_video_model
from src.training.config import Config, load_config

logger = logging.getLogger(__name__)


class SignPredictor:
    """Inference wrapper for sign language recognition.

    Loads a trained model and provides convenient prediction methods
    for both raw video files and precomputed keypoint arrays.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the saved model checkpoint (``.pt`` file).
    cfg : Config
        Configuration used during training.
    device : str
        Device to run inference on (``'cpu'`` or ``'cuda'``).
    class_names : list[str] or None
        Optional list mapping label indices to gloss strings.
        If None, predictions are returned as integer indices.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        cfg: Config,
        device: str = "cpu",
        class_names: Optional[list[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.class_names = class_names

        # Build model
        if cfg.approach in ("pose_transformer", "pose_bilstm"):
            self.model = build_pose_model(cfg)
        elif cfg.approach == "video":
            self.model = build_video_model(cfg)
        else:
            raise ValueError(f"Prediction for approach '{cfg.approach}' not yet supported")

        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Transforms for keypoint preprocessing
        self.transform = get_val_transforms(T=cfg.T)

        logger.info(
            "Loaded model from %s (approach=%s, device=%s)",
            checkpoint_path,
            cfg.approach,
            self.device,
        )

    def predict(self, video_path: str | Path) -> dict:
        """Run prediction on a video file.

        Extracts MediaPipe keypoints from the video, normalizes them,
        and runs the model.

        Parameters
        ----------
        video_path : str or Path
            Path to the input video file.

        Returns
        -------
        dict
            Keys: ``gloss`` (str or int), ``confidence`` (float),
            ``top5`` (list of (gloss, probability) tuples),
            ``label_idx`` (int).
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if self.cfg.approach in ("pose_transformer", "pose_bilstm"):
            # Extract keypoints on the fly
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            keypoints = extract_keypoints_mediapipe(video_path, tmp_path)
            if keypoints is None:
                raise RuntimeError(f"Failed to extract keypoints from {video_path}")

            keypoints = normalize_keypoints(keypoints)
            tmp_path.unlink(missing_ok=True)

            return self._predict_from_keypoints(keypoints)

        elif self.cfg.approach == "video":
            return self._predict_from_video(video_path)

        else:
            raise ValueError(f"Unsupported approach: {self.cfg.approach}")

    def predict_keypoints(self, npy_path: str | Path) -> dict:
        """Run prediction from a saved ``.npy`` keypoint file.

        Parameters
        ----------
        npy_path : str or Path
            Path to a NumPy file with shape ``(T, NUM_KEYPOINTS, 3)``.

        Returns
        -------
        dict
            Same format as ``predict``.
        """
        npy_path = Path(npy_path)
        keypoints = np.load(str(npy_path))
        return self._predict_from_keypoints(keypoints)

    def _predict_from_keypoints(self, keypoints: np.ndarray) -> dict:
        """Core prediction logic for keypoint input."""
        # Apply val transforms (temporal crop to T frames)
        keypoints = self.transform(keypoints)

        # Ensure 3D: (T, K, 3)
        if keypoints.ndim == 2:
            T = keypoints.shape[0]
            keypoints = keypoints.reshape(T, -1, 3)

        # Compute velocity if use_motion is enabled
        if getattr(self.cfg, "use_motion", False):
            velocity = np.zeros_like(keypoints)
            velocity[1:] = keypoints[1:] - keypoints[:-1]
            keypoints = np.concatenate([keypoints, velocity], axis=-1)

        # Flatten: (T, K, C) -> (T, K*C)
        T, K, C = keypoints.shape
        keypoints = keypoints.reshape(T, K * C)

        # To tensor and add batch dim
        tensor = torch.from_numpy(keypoints).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)  # (1, num_classes)
            probs = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

        return self._format_result(probs)

    def _predict_from_video(self, video_path: Path) -> dict:
        """Core prediction logic for raw video input."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV reads BGR; convert to RGB for correct ImageNet normalization
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.cfg.image_size, self.cfg.image_size))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames read from {video_path}")

        frames_arr = np.stack(frames, axis=0)  # (N, H, W, 3)
        # Sample T frames uniformly
        indices = np.linspace(0, len(frames_arr) - 1, self.cfg.T, dtype=np.int64)
        frames_arr = frames_arr[indices]

        # Normalize
        tensor = torch.from_numpy(frames_arr).float() / 255.0
        tensor = tensor.permute(3, 0, 1, 2)  # (3, T, H, W)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, 3, T, H, W)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        return self._format_result(probs)

    def _format_result(self, probs: torch.Tensor) -> dict:
        """Format raw probabilities into a structured result dict."""
        top5_probs, top5_indices = probs.topk(min(5, len(probs)))
        top5_probs = top5_probs.cpu().numpy()
        top5_indices = top5_indices.cpu().numpy()

        pred_idx = int(top5_indices[0])
        confidence = float(top5_probs[0])

        if self.class_names is not None and pred_idx < len(self.class_names):
            gloss = self.class_names[pred_idx]
            top5 = [
                (self.class_names[int(idx)] if int(idx) < len(self.class_names) else str(idx), float(p))
                for idx, p in zip(top5_indices, top5_probs)
            ]
        else:
            gloss = str(pred_idx)
            top5 = [(str(int(idx)), float(p)) for idx, p in zip(top5_indices, top5_probs)]

        return {
            "gloss": gloss,
            "confidence": confidence,
            "label_idx": pred_idx,
            "top5": top5,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_class_names(cfg: Config) -> list[str]:
    """Attempt to load class names from the training split CSV."""
    import pandas as pd

    data_dir = Path(cfg.data_dir)
    for split in ["train", "val", "test"]:
        csv_path = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}" / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            names = [""] * cfg.num_classes
            for _, row in df.iterrows():
                idx = int(row["label_idx"])
                if idx < cfg.num_classes:
                    names[idx] = row["gloss"]
            return names
    return [str(i) for i in range(cfg.num_classes)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sign from video")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--keypoints", type=str, help="Path to .npy keypoint file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    class_names = _load_class_names(cfg)
    predictor = SignPredictor(
        checkpoint_path=args.checkpoint,
        cfg=cfg,
        device=args.device,
        class_names=class_names,
    )

    if args.video:
        result = predictor.predict(args.video)
    elif args.keypoints:
        result = predictor.predict_keypoints(args.keypoints)
    else:
        parser.error("Provide either --video or --keypoints")

    print(f"\nPrediction: {result['gloss']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nTop-5 predictions:")
    for gloss, prob in result["top5"]:
        print(f"  {gloss:20s}  {prob:.4f}")
