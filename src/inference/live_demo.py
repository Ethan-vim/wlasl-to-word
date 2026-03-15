"""
Real-time webcam inference for ASL sign language recognition.

This is the main deliverable of the project.  It opens a webcam feed,
runs MediaPipe Holistic on each frame to extract keypoints, buffers a
rolling window of T frames, runs the trained model every inference
interval, and displays the predicted sign with confidence overlaid on
the video feed.

Architecture:
    - Capture thread: reads webcam frames at full speed
    - Inference thread: runs model on the buffered keypoints periodically
    - Main thread: renders the display overlay and handles user input

Press 'q' to quit.  Press 's' to save the current prediction to a log.
"""

import argparse
import collections
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.data.preprocess import (
    NUM_KEYPOINTS,
    _import_mediapipe_drawing,
    _import_mediapipe_holistic,
    normalize_keypoints,
)
from src.models.prototypical import build_model
from src.training.config import Config, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame buffer
# ---------------------------------------------------------------------------


class FrameBuffer:
    """Thread-safe fixed-size deque for storing recent keypoint frames.

    Parameters
    ----------
    max_size : int
        Maximum number of frames to retain.
    """

    def __init__(self, max_size: int = 64) -> None:
        self.max_size = max_size
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=max_size)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        """Add a keypoint frame to the buffer (thread-safe)."""
        with self._lock:
            self._buffer.append(frame)

    def get_all(self) -> np.ndarray:
        """Return all buffered frames as a single NumPy array.

        Returns
        -------
        np.ndarray
            Shape ``(N, NUM_KEYPOINTS, 3)`` where N <= max_size.
            Returns an empty array if buffer is empty.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32)
            return np.stack(list(self._buffer), axis=0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Remove all frames from the buffer."""
        with self._lock:
            self._buffer.clear()


# ---------------------------------------------------------------------------
# Live predictor
# ---------------------------------------------------------------------------


class LivePredictor:
    """Manages the model and performs inference on buffered keypoint sequences.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the model checkpoint.
    cfg : Config
        Configuration.
    device : str
        Device for inference.
    class_names : list[str] or None
        Gloss names indexed by label.
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
        self.class_names = class_names or [str(i) for i in range(cfg.num_classes)]

        # Build prototypical model
        self.model = build_model(cfg)
        ckpt = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False
        )
        state_dict = ckpt["model_state_dict"]
        # Remove prototypes from checkpoint — they'll be recomputed from training data
        state_dict.pop("prototypes", None)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Compute prototypes from training data
        self._load_prototypes(cfg)

        # MediaPipe — uses shared helper that handles Windows/Python 3.12 fallback
        self._mp_holistic = _import_mediapipe_holistic()
        holistic_mod, drawing_mod, styles_mod = _import_mediapipe_drawing()
        self._mp_drawing = drawing_mod
        self._mp_drawing_styles = styles_mod
        self.holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            refine_face_landmarks=True,
        )

        logger.info("LivePredictor initialized (device=%s)", self.device)

    def _load_prototypes(self, cfg: Config) -> None:
        """Compute prototypes from training data for inference."""
        from src.data.augment import get_val_transforms
        from src.data.dataset import WLASLKeypointDataset, get_dataloader

        data_dir = Path(cfg.data_dir)
        train_csv = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}" / "train.csv"
        processed_dir = data_dir / "processed"

        if not train_csv.exists():
            logger.warning("Training split not found; prototypes not computed.")
            return

        transform = get_val_transforms(T=cfg.T)
        train_ds = WLASLKeypointDataset(
            split_csv=train_csv,
            keypoint_dir=processed_dir,
            transform=transform,
            T=cfg.T,
            use_motion=getattr(cfg, "use_motion", False),
        )
        loader = get_dataloader(train_ds, batch_size=32, shuffle=False, num_workers=0)
        self.model.compute_prototypes(loader)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, object]:
        """Extract MediaPipe keypoints from a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR frame from OpenCV.

        Returns
        -------
        tuple[np.ndarray, object]
            Keypoint array of shape ``(NUM_KEYPOINTS, 3)`` and the
            MediaPipe results object (for drawing landmarks).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        kps = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        offset = 0

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                kps[offset + i] = [lm.x, lm.y, lm.z]
        offset += 33

        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                kps[offset + i] = [lm.x, lm.y, lm.z]
        offset += 21

        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                kps[offset + i] = [lm.x, lm.y, lm.z]
        offset += 21

        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                kps[offset + i] = [lm.x, lm.y, lm.z]

        return kps, results

    def predict_buffer(self, buffer: FrameBuffer) -> Optional[dict]:
        """Run model inference on the current buffer contents.

        Parameters
        ----------
        buffer : FrameBuffer
            The rolling keypoint buffer.

        Returns
        -------
        dict or None
            Prediction result or None if buffer is too short.
        """
        keypoints = buffer.get_all()  # (N, 543, 3)

        if keypoints.shape[0] < 5:
            return None

        # Normalize
        keypoints = normalize_keypoints(keypoints)

        # Pad/crop to T frames
        T = self.cfg.T
        N = keypoints.shape[0]
        if N < T:
            pad = np.tile(keypoints[-1:], (T - N, 1, 1))
            keypoints = np.concatenate([keypoints, pad], axis=0)
        elif N > T:
            indices = np.linspace(0, N - 1, T, dtype=np.int64)
            keypoints = keypoints[indices]

        # Compute velocity if use_motion is enabled
        if getattr(self.cfg, "use_motion", False):
            velocity = np.zeros_like(keypoints)
            velocity[1:] = keypoints[1:] - keypoints[:-1]
            keypoints = np.concatenate([keypoints, velocity], axis=-1)  # (T, 543, 6)

        # Flatten and convert to tensor
        keypoints_flat = keypoints.reshape(T, -1)  # (T, 543*C)
        tensor = torch.from_numpy(keypoints_flat).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model.classify(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        top5_probs, top5_indices = probs.topk(5)
        top5_probs = top5_probs.cpu().numpy()
        top5_indices = top5_indices.cpu().numpy()

        pred_idx = int(top5_indices[0])
        confidence = float(top5_probs[0])
        gloss = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)
        top5 = [
            (self.class_names[int(i)] if int(i) < len(self.class_names) else str(i), float(p))
            for i, p in zip(top5_indices, top5_probs)
        ]

        return {
            "gloss": gloss,
            "confidence": confidence,
            "label_idx": pred_idx,
            "top5": top5,
        }

    @staticmethod
    def smooth_predictions(
        recent_preds: list[dict],
        mode: str = "avg",
    ) -> Optional[dict]:
        """Smooth recent predictions to reduce flickering.

        Parameters
        ----------
        recent_preds : list[dict]
            List of recent prediction dicts.
        mode : str
            ``'avg'`` for probability averaging, ``'majority'`` for
            majority vote.

        Returns
        -------
        dict or None
            Smoothed prediction.
        """
        if not recent_preds:
            return None

        if mode == "majority":
            # Majority vote on the top-1 prediction
            votes = [p["gloss"] for p in recent_preds]
            counter = collections.Counter(votes)
            winner, count = counter.most_common(1)[0]
            # Find the most recent prediction with this gloss for full info
            for p in reversed(recent_preds):
                if p["gloss"] == winner:
                    return {**p, "confidence": count / len(votes)}
            return recent_preds[-1]

        else:  # avg
            # Average the top-5 probabilities across all prediction windows.
            # Glosses absent from a window's top-5 are treated as 0 probability,
            # so we divide by the total number of windows (not just appearances).
            n_windows = len(recent_preds)
            gloss_probs: dict[str, float] = {}
            for p in recent_preds:
                for gloss, prob in p["top5"]:
                    gloss_probs[gloss] = gloss_probs.get(gloss, 0.0) + prob

            # Compute mean probability for each gloss
            avg_probs = {g: total / n_windows for g, total in gloss_probs.items()}
            sorted_glosses = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)

            top5 = sorted_glosses[:5]
            winner = top5[0][0]
            confidence = top5[0][1]

            # Find label_idx from the most recent prediction
            label_idx = recent_preds[-1].get("label_idx", 0)
            for p in recent_preds:
                if p["gloss"] == winner:
                    label_idx = p["label_idx"]
                    break

            return {
                "gloss": winner,
                "confidence": float(confidence),
                "label_idx": label_idx,
                "top5": [(g, float(p)) for g, p in top5],
            }


# ---------------------------------------------------------------------------
# Display overlay
# ---------------------------------------------------------------------------


class ASLDisplay:
    """Handles drawing prediction overlays on the webcam feed."""

    # Colors (BGR)
    BG_COLOR = (40, 40, 40)
    TEXT_COLOR = (255, 255, 255)
    ACCENT_COLOR = (0, 200, 100)
    CONF_BAR_BG = (80, 80, 80)
    CONF_BAR_FG = (0, 200, 100)
    LOW_CONF_FG = (0, 100, 200)

    @staticmethod
    def draw_overlay(
        frame: np.ndarray,
        prediction: Optional[dict],
        confidence: float,
        top5: Optional[list[tuple[str, float]]],
        mp_results: Optional[object] = None,
    ) -> np.ndarray:
        """Draw the prediction overlay on a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on (modified in place).
        prediction : dict or None
            Current prediction dict.
        confidence : float
            Confidence score for display.
        top5 : list or None
            Top-5 predictions.
        mp_results : object or None
            MediaPipe results for drawing landmarks.

        Returns
        -------
        np.ndarray
            The annotated frame.
        """
        h, w = frame.shape[:2]

        # Draw MediaPipe landmarks if available
        if mp_results is not None:
            mp_holistic, mp_drawing, _ = _import_mediapipe_drawing()
            if mp_drawing is None:
                return frame  # Skip landmark drawing if mediapipe unavailable
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            if mp_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    mp_results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(100, 200, 100), thickness=1
                    ),
                )
            if mp_results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    mp_results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(200, 100, 100), thickness=2
                    ),
                )
            if mp_results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    mp_results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(100, 100, 200), thickness=2
                    ),
                )

        # Semi-transparent overlay panel at the top
        overlay = frame.copy()
        panel_h = 120
        cv2.rectangle(overlay, (0, 0), (w, panel_h), ASLDisplay.BG_COLOR, -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        if prediction is not None and confidence > 0:
            gloss = prediction.get("gloss", "---")

            # Main prediction text
            cv2.putText(
                frame,
                gloss.upper(),
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                ASLDisplay.ACCENT_COLOR,
                3,
                cv2.LINE_AA,
            )

            # Confidence bar
            bar_x, bar_y = 20, 70
            bar_w, bar_h = 200, 20
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_w, bar_y + bar_h),
                ASLDisplay.CONF_BAR_BG,
                -1,
            )
            fill_w = int(bar_w * confidence)
            bar_color = ASLDisplay.CONF_BAR_FG if confidence > 0.6 else ASLDisplay.LOW_CONF_FG
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_w, bar_y + bar_h),
                bar_color,
                -1,
            )
            cv2.putText(
                frame,
                f"{confidence:.0%}",
                (bar_x + bar_w + 10, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ASLDisplay.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

            # Top-5 predictions (right side)
            if top5 is not None:
                x_start = w - 280
                for i, (g, p) in enumerate(top5):
                    y_pos = 25 + i * 20
                    text = f"{i + 1}. {g}: {p:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (x_start, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        ASLDisplay.TEXT_COLOR,
                        1,
                        cv2.LINE_AA,
                    )
        else:
            cv2.putText(
                frame,
                "Waiting for sign...",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (150, 150, 150),
                2,
                cv2.LINE_AA,
            )

        return frame


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------


def run_demo(
    cfg: Config,
    checkpoint_path: str | Path,
    camera_id: int = 0,
    device: str = "cpu",
) -> None:
    """Run the real-time ASL recognition demo.

    Opens a webcam, extracts keypoints per frame, buffers them,
    and periodically runs inference with smooth display updates.

    Parameters
    ----------
    cfg : Config
        Configuration.
    checkpoint_path : str or Path
        Path to the model checkpoint.
    camera_id : int
        OpenCV camera device ID.
    device : str
        Inference device.
    """
    # Load class names
    import pandas as pd

    class_names = [str(i) for i in range(cfg.num_classes)]
    data_dir = Path(cfg.data_dir)
    for split in ["train", "val"]:
        csv_path = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}" / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                idx = int(row["label_idx"])
                if idx < cfg.num_classes:
                    class_names[idx] = row["gloss"]
            break

    # Initialize components
    predictor = LivePredictor(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        device=device,
        class_names=class_names,
    )
    buffer = FrameBuffer(max_size=cfg.buffer_size)
    display = ASLDisplay()

    # State
    current_prediction: Optional[dict] = None
    current_confidence: float = 0.0
    current_top5: Optional[list] = None
    recent_predictions: list[dict] = []
    last_mp_results: Optional[object] = None
    inference_lock = threading.Lock()
    running = True
    saved_predictions: list[str] = []

    # FPS tracking
    frame_times: collections.deque[float] = collections.deque(maxlen=30)

    # Inference thread
    def inference_loop() -> None:
        nonlocal current_prediction, current_confidence, current_top5, recent_predictions
        while running:
            time.sleep(0.5)  # Run inference every 0.5 seconds
            if not running:
                break

            result = predictor.predict_buffer(buffer)
            if result is None:
                continue

            with inference_lock:
                recent_predictions.append(result)
                # Keep only the last N predictions for smoothing
                if len(recent_predictions) > cfg.smoothing_window:
                    recent_predictions = recent_predictions[-cfg.smoothing_window:]

                smoothed = predictor.smooth_predictions(recent_predictions, mode="avg")
                if smoothed is not None and smoothed["confidence"] >= cfg.confidence_threshold:
                    current_prediction = smoothed
                    current_confidence = smoothed["confidence"]
                    current_top5 = smoothed.get("top5")
                else:
                    current_prediction = None
                    current_confidence = 0.0
                    current_top5 = None

    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    inference_thread.start()

    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", camera_id)
        running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("ASL Recognition Demo started. Press 'q' to quit, 's' to save prediction.")

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break

            t0 = time.perf_counter()

            # Extract keypoints
            kps, mp_results = predictor.preprocess_frame(frame)
            buffer.push(kps)
            last_mp_results = mp_results

            # Get current prediction (thread-safe read)
            with inference_lock:
                pred_copy = current_prediction
                conf_copy = current_confidence
                top5_copy = current_top5

            # Draw overlay
            frame = display.draw_overlay(
                frame,
                pred_copy,
                conf_copy,
                top5_copy,
                last_mp_results,
            )

            # FPS counter
            t1 = time.perf_counter()
            frame_times.append(t1 - t0)
            if cfg.fps_display and len(frame_times) > 1:
                avg_time = np.mean(list(frame_times))
                fps = 1.0 / max(avg_time, 1e-6)
                cv2.putText(
                    frame,
                    f"FPS: {fps:.0f}",
                    (frame.shape[1] - 120, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Buffer status
            buf_len = len(buffer)
            cv2.putText(
                frame,
                f"Buffer: {buf_len}/{cfg.buffer_size}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("ASL Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if pred_copy is not None:
                    entry = f"{pred_copy['gloss']} ({conf_copy:.2f})"
                    saved_predictions.append(entry)
                    print(f"Saved: {entry}")

    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()
        predictor.holistic.close()

        if saved_predictions:
            print("\nSaved predictions:")
            for entry in saved_predictions:
                print(f"  - {entry}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live ASL Recognition Demo")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
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
    run_demo(cfg, args.checkpoint, camera_id=args.camera, device=args.device)
