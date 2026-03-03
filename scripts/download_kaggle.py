"""
Download WLASL videos from Kaggle (faster alternative to URL-based downloading).

This script uses the Kaggle API to download the WLASL video dataset and
set up the data directory. It is significantly faster than the URL-based
download approach since Kaggle hosts the full archive (~5 GB, ~12K videos).

Prerequisites:
    1. Install the kaggle package: pip install kaggle
    2. Set up Kaggle API credentials:
       - Go to https://www.kaggle.com/settings → Create New Token
       - This downloads kaggle.json — move it to ~/.kaggle/kaggle.json
       - chmod 600 ~/.kaggle/kaggle.json

Usage:
    python scripts/download_kaggle.py
    python scripts/download_kaggle.py --data-dir data --subset WLASL100
    python scripts/download_kaggle.py --dataset risangbaskoro/wlasl-processed
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (
    download_wlasl_annotations,
    parse_wlasl_annotations,
    WLASL_VARIANT_SIZES,
)

logger = logging.getLogger(__name__)

KAGGLE_DATASET_SLUG = "risangbaskoro/wlasl-processed"


def _check_kaggle_available() -> bool:
    """Check if the kaggle package is installed and credentials are configured."""
    try:
        import kaggle  # noqa: F401

        return True
    except ImportError:
        logger.error(
            "The 'kaggle' package is not installed. Install it with:\n"
            "  pip install kaggle"
        )
        return False
    except (OSError, ValueError) as e:
        # kaggle raises OSError if credentials file is missing,
        # ValueError if credentials file exists but is malformed
        # (e.g. missing username/key fields)
        logger.error(
            "Kaggle API credentials not found or invalid. Set up credentials:\n"
            "  1. Go to https://www.kaggle.com/settings → Create New Token\n"
            "  2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json\n"
            "  3. Run: chmod 600 ~/.kaggle/kaggle.json\n"
            "  The file should contain: {\"username\":\"YOUR_USER\",\"key\":\"YOUR_KEY\"}\n\n"
            "  Original error: %s",
            e,
        )
        return False


def download_from_kaggle(
    data_dir: Path,
    dataset_slug: str = KAGGLE_DATASET_SLUG,
) -> Path:
    """Download and extract the WLASL dataset from Kaggle.

    Parameters
    ----------
    data_dir : Path
        Root data directory (e.g., ``data/``).
    dataset_slug : str
        Kaggle dataset identifier (default: ``risangbaskoro/wlasl-processed``).

    Returns
    -------
    Path
        Path to the ``data/raw/`` directory containing the extracted videos.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Use a temp download dir so we can move files into raw/
    download_dir = data_dir / "_kaggle_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset '%s' from Kaggle...", dataset_slug)
    logger.info("This is ~5 GB and may take several minutes.")

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        dataset_slug,
        path=str(download_dir),
        unzip=True,
    )

    logger.info("Download complete. Moving videos to %s", raw_dir)

    # The dataset may contain videos directly or in a videos/ subfolder.
    # Also look for the annotation JSON.
    video_count = 0
    for item in download_dir.rglob("*"):
        if item.is_file():
            if item.suffix.lower() == ".mp4":
                dest = raw_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
                video_count += 1
            elif item.name == "WLASL_v0.3.json":
                ann_dir = data_dir / "annotations"
                ann_dir.mkdir(parents=True, exist_ok=True)
                dest = ann_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
                logger.info("Found annotation JSON: %s", dest)

    # Clean up temp download directory
    shutil.rmtree(download_dir, ignore_errors=True)

    logger.info("Moved %d video files to %s", video_count, raw_dir)
    return raw_dir


def main() -> None:
    """Download WLASL videos from Kaggle and set up the data directory."""
    parser = argparse.ArgumentParser(
        description="Download WLASL videos from Kaggle (fast alternative)"
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
        help="WLASL variant for annotation summary (default: WLASL100)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=KAGGLE_DATASET_SLUG,
        help=f"Kaggle dataset slug (default: {KAGGLE_DATASET_SLUG})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)

    # Check kaggle is available before proceeding
    if not _check_kaggle_available():
        sys.exit(1)

    # Create directory structure
    for d in ["raw", "processed", "annotations", "splits"]:
        (data_dir / d).mkdir(parents=True, exist_ok=True)

    # Download videos from Kaggle
    raw_dir = download_from_kaggle(data_dir, dataset_slug=args.dataset)

    # Count what we got
    video_files = list(raw_dir.glob("*.mp4"))
    print(f"\n  Videos in {raw_dir}: {len(video_files)}")

    # Download annotations (if not already provided by the Kaggle dataset)
    ann_path = data_dir / "annotations" / "WLASL_v0.3.json"
    if not ann_path.exists():
        logger.info("Annotation JSON not found in Kaggle download, fetching from GitHub...")
        ann_path = download_wlasl_annotations(data_dir / "annotations")

    # Parse and show statistics
    df = parse_wlasl_annotations(ann_path, subset=args.subset)

    print("\n" + "=" * 70)
    print(f"  WLASL Dataset Summary ({args.subset})")
    print("=" * 70)
    print(f"  Total instances (annotation) : {len(df)}")
    print(f"  Unique glosses               : {df['gloss'].nunique()}")
    print(f"  Videos downloaded             : {len(video_files)}")

    # Count how many annotated videos we actually have
    annotated_ids = set(df["video_id"].astype(str))
    downloaded_ids = {f.stem for f in video_files}
    matched = annotated_ids & downloaded_ids
    print(f"  Matched (annotation + video)  : {len(matched)}")

    print(f"\n  Train samples : {len(df[df['split'] == 'train'])}")
    print(f"  Val samples   : {len(df[df['split'] == 'val'])}")
    print(f"  Test samples  : {len(df[df['split'] == 'test'])}")

    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print(f"""
  1. (Recommended) Validate downloaded videos:
       python scripts/validate_videos.py --video-dir {raw_dir} --delete

  2. Preprocess keypoints:
       python -m src.data.preprocess --data-dir {data_dir} --subset {args.subset}

  3. Train:
       python -m src.training.train --config configs/pose_transformer.yaml
""")


if __name__ == "__main__":
    main()
