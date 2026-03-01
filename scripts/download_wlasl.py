"""
Script to download WLASL dataset annotations and set up the data directory.

This script downloads the annotation JSON file from the official WLASL
GitHub repository and prints instructions for obtaining the video files,
which require following the repository's download procedures.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (
    download_wlasl_annotations,
    parse_wlasl_annotations,
    WLASL_VARIANT_SIZES,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Download annotations and set up the data directory structure."""
    parser = argparse.ArgumentParser(
        description="Download WLASL annotations and set up data directories"
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
        help="WLASL variant (default: WLASL100)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)

    # Create directory structure
    dirs = [
        data_dir / "raw",
        data_dir / "processed",
        data_dir / "annotations",
        data_dir / "splits",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", d)

    # Download annotations
    annotation_path = download_wlasl_annotations(data_dir / "annotations")

    # Parse and show statistics
    df = parse_wlasl_annotations(annotation_path, subset=args.subset)

    print("\n" + "=" * 70)
    print(f"  WLASL Annotation Summary ({args.subset})")
    print("=" * 70)
    print(f"  Total instances : {len(df)}")
    print(f"  Unique glosses  : {df['gloss'].nunique()}")
    print(f"  Train samples   : {len(df[df['split'] == 'train'])}")
    print(f"  Val samples     : {len(df[df['split'] == 'val'])}")
    print(f"  Test samples    : {len(df[df['split'] == 'test'])}")
    print(f"  Unique signers  : {df['signer_id'].nunique()}")
    print()
    print("  Top 10 glosses by sample count:")
    top_glosses = df["gloss"].value_counts().head(10)
    for gloss, count in top_glosses.items():
        print(f"    {gloss:20s}  {count} samples")

    print("\n" + "=" * 70)
    print("  HOW TO DOWNLOAD VIDEOS")
    print("=" * 70)
    print("""
  The WLASL video files are NOT distributed directly with the annotations.
  To obtain the videos, follow one of these approaches:

  1. OFFICIAL REPOSITORY SCRIPTS
     Clone the WLASL repo and use their download scripts:
       git clone https://github.com/dxli94/WLASL.git
       cd WLASL/start_kit
       python video_downloader.py

     Note: Many original URLs have expired. The download script handles
     retries but some videos may be unavailable.

  2. COMMUNITY MIRRORS
     Several community members have archived the dataset. Search for
     "WLASL dataset download" on Kaggle or academic dataset repositories.
     Common mirrors:
       - Kaggle datasets (search for "WLASL")
       - Google Drive mirrors referenced in GitHub issues

  3. PARTIAL DATASET
     You can work with whatever subset of videos you can obtain.
     The preprocessing pipeline automatically filters to only the
     available videos.

  AFTER DOWNLOADING:
    Place all video files (*.mp4) in:
      {raw_dir}

    Then run the preprocessing pipeline:
      python -m src.data.preprocess --data-dir {data_dir} --subset {subset}

    Splits will be saved to:
      {splits_dir}
""".format(
        raw_dir=data_dir / "raw",
        data_dir=data_dir,
        subset=args.subset,
        splits_dir=data_dir / "splits" / args.subset,
    ))

    # Save the parsed annotation DataFrame for reference
    csv_path = data_dir / "annotations" / f"{args.subset}_annotations.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved annotation CSV to: {csv_path}")
    print()


if __name__ == "__main__":
    main()
