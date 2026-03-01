"""
Comprehensive preprocessing module for the WLASL dataset.

Handles downloading annotations, parsing them into structured DataFrames,
extracting video frames, computing MediaPipe keypoints, normalizing
keypoint coordinates, and orchestrating full dataset preprocessing.
"""

import argparse
import json
import logging
import multiprocessing
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WLASL_ANNOTATION_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)

# MediaPipe Holistic landmark counts
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS = 468
NUM_KEYPOINTS = NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS * 2 + NUM_FACE_LANDMARKS  # 543

WLASL_VARIANT_SIZES = {
    "WLASL100": 100,
    "WLASL300": 300,
    "WLASL1000": 1000,
    "WLASL2000": 2000,
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_wlasl_annotations(save_dir: str | Path) -> Path:
    """Download the WLASL JSON annotation file from the official GitHub repo.

    Parameters
    ----------
    save_dir : str or Path
        Directory to save the downloaded JSON file.

    Returns
    -------
    Path
        Path to the saved annotation file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dest = save_dir / "WLASL_v0.3.json"

    if dest.exists():
        logger.info("Annotation file already exists at %s", dest)
        return dest

    logger.info("Downloading WLASL annotations from %s", WLASL_ANNOTATION_URL)
    urllib.request.urlretrieve(WLASL_ANNOTATION_URL, str(dest))
    logger.info("Saved annotations to %s", dest)
    return dest


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def parse_wlasl_annotations(
    json_path: str | Path,
    subset: str = "WLASL100",
) -> pd.DataFrame:
    """Parse the WLASL annotation JSON into a structured DataFrame.

    The JSON is a list of gloss entries. Each gloss entry has a ``gloss``
    string and an ``instances`` list.  This function selects the top-N
    glosses (sorted by their order in the JSON, which is the canonical
    ordering used by the authors) to match the requested WLASL variant.

    Parameters
    ----------
    json_path : str or Path
        Path to the WLASL JSON annotation file.
    subset : str
        One of ``'WLASL100'``, ``'WLASL300'``, ``'WLASL1000'``, ``'WLASL2000'``.

    Returns
    -------
    pd.DataFrame
        Columns: video_id, gloss, label_idx, split, signer_id, fps, bbox, url
    """
    json_path = Path(json_path)
    if subset not in WLASL_VARIANT_SIZES:
        raise ValueError(
            f"Unknown subset '{subset}'. Choose from {list(WLASL_VARIANT_SIZES)}"
        )

    num_glosses = WLASL_VARIANT_SIZES[subset]

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Take only the first `num_glosses` entries (canonical ordering)
    data = data[:num_glosses]

    rows: list[dict] = []
    for label_idx, entry in enumerate(data):
        gloss = entry["gloss"]
        for inst in entry.get("instances", []):
            split_map = {"train": "train", "val": "val", "test": "test"}
            split = split_map.get(inst.get("split", "train"), "train")
            bbox = inst.get("bbox", None)
            # bbox may be a list of four ints or None
            if bbox is not None:
                bbox = list(bbox)
            rows.append(
                {
                    "video_id": inst.get("video_id", ""),
                    "gloss": gloss,
                    "label_idx": label_idx,
                    "split": split,
                    "signer_id": inst.get("signer_id", -1),
                    "fps": inst.get("fps", 25),
                    "bbox": bbox,
                    "url": inst.get("url", ""),
                }
            )

    df = pd.DataFrame(rows)
    logger.info(
        "Parsed %d instances across %d glosses for %s",
        len(df),
        num_glosses,
        subset,
    )
    return df


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: int = 25,
) -> int:
    """Extract frames from a video at a given FPS and save as JPEG images.

    Parameters
    ----------
    video_path : str or Path
        Path to the input video file.
    output_dir : str or Path
        Directory to save extracted frame images.
    fps : int
        Target frames-per-second for extraction.

    Returns
    -------
    int
        Number of frames extracted.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0

    frame_interval = max(1, round(src_fps / fps))
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            out_path = output_dir / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    return saved_count


# ---------------------------------------------------------------------------
# Keypoint extraction
# ---------------------------------------------------------------------------


def extract_keypoints_mediapipe(
    video_path: str | Path,
    output_path: str | Path,
) -> Optional[np.ndarray]:
    """Run MediaPipe Holistic on every frame and save keypoints as .npy.

    The output array has shape ``(T, NUM_KEYPOINTS, 3)`` where
    ``NUM_KEYPOINTS = 543`` (33 pose + 21 left hand + 21 right hand + 468 face).
    Missing detections are zero-padded.

    Parameters
    ----------
    video_path : str or Path
        Path to the input video.
    output_path : str or Path
        Destination ``.npy`` file path.

    Returns
    -------
    np.ndarray or None
        The keypoint array, or None if the video could not be opened.
    """
    # Lazy import to avoid loading mediapipe at module level
    import mediapipe as mp

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return None

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_keypoints: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        frame_kps = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

        offset = 0
        # Pose landmarks (33)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_kps[offset + i] = [lm.x, lm.y, lm.z]
        offset += NUM_POSE_LANDMARKS

        # Left hand landmarks (21)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                frame_kps[offset + i] = [lm.x, lm.y, lm.z]
        offset += NUM_HAND_LANDMARKS

        # Right hand landmarks (21)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                frame_kps[offset + i] = [lm.x, lm.y, lm.z]
        offset += NUM_HAND_LANDMARKS

        # Face landmarks (468)
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                frame_kps[offset + i] = [lm.x, lm.y, lm.z]

        all_keypoints.append(frame_kps)

    cap.release()
    holistic.close()

    if len(all_keypoints) == 0:
        logger.warning("No frames read from video: %s", video_path)
        return None

    keypoints = np.stack(all_keypoints, axis=0)  # (T, 543, 3)
    np.save(str(output_path), keypoints)
    return keypoints


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Normalize keypoint coordinates relative to the nose and shoulder width.

    1. Translate so pose landmark 0 (nose) is at the origin for every frame.
    2. Scale so the shoulder width (distance between landmarks 11 and 12) is 1.

    Parameters
    ----------
    keypoints : np.ndarray
        Shape ``(T, NUM_KEYPOINTS, 3)``.

    Returns
    -------
    np.ndarray
        Normalized keypoints with the same shape.
    """
    kps = keypoints.copy()
    T = kps.shape[0]

    # Nose is pose index 0 (first landmark overall)
    nose = kps[:, 0:1, :]  # (T, 1, 3)
    kps = kps - nose  # translate to nose-centered

    # Shoulder width: pose index 11 (left shoulder) and 12 (right shoulder)
    left_shoulder = kps[:, 11, :]  # (T, 3)
    right_shoulder = kps[:, 12, :]  # (T, 3)
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1)  # (T,)

    # Avoid division by zero: use the median non-zero shoulder width as scale
    valid_widths = shoulder_width[shoulder_width > 1e-6]
    if len(valid_widths) == 0:
        scale = 1.0
    else:
        scale = float(np.median(valid_widths))

    kps = kps / max(scale, 1e-6)
    return kps


# ---------------------------------------------------------------------------
# Dataset orchestration
# ---------------------------------------------------------------------------


def _process_single_video(args: tuple) -> Optional[str]:
    """Worker function for multiprocessing keypoint extraction.

    Parameters
    ----------
    args : tuple
        (video_id, video_path, output_path)

    Returns
    -------
    str or None
        video_id on success, None on failure.
    """
    video_id, video_path, output_path = args
    video_path = Path(video_path)
    output_path = Path(output_path)

    if output_path.exists():
        return video_id

    if not video_path.exists():
        return None

    result = extract_keypoints_mediapipe(video_path, output_path)
    if result is None:
        return None

    # Also normalize and overwrite
    normalized = normalize_keypoints(result)
    np.save(str(output_path), normalized)
    return video_id


def preprocess_dataset(
    annotation_df: pd.DataFrame,
    video_dir: str | Path,
    output_dir: str | Path,
    mode: str = "keypoints",
    max_workers: int = 4,
) -> pd.DataFrame:
    """Orchestrate full preprocessing for all videos in the annotation DataFrame.

    Parameters
    ----------
    annotation_df : pd.DataFrame
        DataFrame from ``parse_wlasl_annotations``.
    video_dir : str or Path
        Directory containing raw video files.
    output_dir : str or Path
        Directory to write processed outputs.
    mode : str
        ``'keypoints'`` to extract and save ``.npy`` keypoint files,
        ``'frames'`` to extract JPEG frames.
    max_workers : int
        Number of parallel workers for processing.

    Returns
    -------
    pd.DataFrame
        Updated annotation DataFrame filtered to only successfully processed videos.
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "keypoints":
        tasks = []
        for _, row in annotation_df.iterrows():
            vid = row["video_id"]
            # Try common extensions
            video_path = None
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                candidate = video_dir / f"{vid}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            if video_path is None:
                video_path = video_dir / f"{vid}.mp4"

            out_path = output_dir / f"{vid}.npy"
            tasks.append((vid, str(video_path), str(out_path)))

        successful_ids = set()
        # Use "spawn" context to avoid fork + MediaPipe crashes on macOS
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(_process_single_video, t): t[0] for t in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Extracting keypoints"
            ):
                result = future.result()
                if result is not None:
                    successful_ids.add(result)

        logger.info(
            "Successfully processed %d / %d videos",
            len(successful_ids),
            len(annotation_df),
        )
        return annotation_df[annotation_df["video_id"].isin(successful_ids)].copy()

    elif mode == "frames":
        successful_ids = set()
        for _, row in tqdm(
            annotation_df.iterrows(),
            total=len(annotation_df),
            desc="Extracting frames",
        ):
            vid = row["video_id"]
            video_path = None
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                candidate = video_dir / f"{vid}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            if video_path is None:
                continue

            frame_dir = output_dir / vid
            n_frames = extract_frames(video_path, frame_dir, fps=25)
            if n_frames > 0:
                successful_ids.add(vid)

        logger.info(
            "Successfully extracted frames for %d / %d videos",
            len(successful_ids),
            len(annotation_df),
        )
        return annotation_df[annotation_df["video_id"].isin(successful_ids)].copy()

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'keypoints' or 'frames'.")


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------


def create_splits(
    annotation_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save train/val/test CSVs using the official WLASL splits.

    Parameters
    ----------
    annotation_df : pd.DataFrame
        DataFrame with a ``split`` column containing 'train', 'val', or 'test'.
    output_dir : str or Path
        Directory to write the split CSV files.

    Returns
    -------
    dict[str, Path]
        Mapping from split name to the saved CSV path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for split_name in ["train", "val", "test"]:
        split_df = annotation_df[annotation_df["split"] == split_name]
        out_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        paths[split_name] = out_path
        logger.info(
            "Saved %s split: %d samples -> %s", split_name, len(split_df), out_path
        )

    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for data preprocessing."""
    parser = argparse.ArgumentParser(
        description="WLASL data preprocessing pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="WLASL100",
        choices=list(WLASL_VARIANT_SIZES),
        help="WLASL variant to use (default: WLASL100)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="keypoints",
        choices=["keypoints", "frames"],
        help="Preprocessing mode (default: keypoints)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)

    # Step 1: Download annotations
    annotation_path = download_wlasl_annotations(data_dir / "annotations")

    # Step 2: Parse annotations
    df = parse_wlasl_annotations(annotation_path, subset=args.subset)

    # Step 3: Preprocess videos
    video_dir = data_dir / "raw"
    output_dir = data_dir / "processed"
    df = preprocess_dataset(
        df,
        video_dir=video_dir,
        output_dir=output_dir,
        mode=args.mode,
        max_workers=args.max_workers,
    )

    # Step 4: Create splits (stored under a variant-specific subdirectory so
    # multiple WLASL variants can coexist without overwriting each other)
    create_splits(df, data_dir / "splits" / args.subset)

    print(f"\nPreprocessing complete. Processed {len(df)} videos.")
    print(f"Splits saved to {data_dir / 'splits'}")


if __name__ == "__main__":
    main()
