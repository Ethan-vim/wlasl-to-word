"""
Scan a directory of downloaded WLASL videos and identify files that are
HTML redirect pages masquerading as .mp4 files (a common issue with expired
WLASL download URLs).

Usage
-----
    # Report only
    python scripts/validate_videos.py --video-dir WLASL/start_kit/raw_videos

    # Report and delete the bad files
    python scripts/validate_videos.py --video-dir WLASL/start_kit/raw_videos --delete

    # Also copy valid video IDs to a text file for reference
    python scripts/validate_videos.py --video-dir WLASL/start_kit/raw_videos --save-valid valid_ids.txt
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# HTML signatures that indicate a redirect/lander page saved as a video file
_HTML_SIGNATURES: list[bytes] = [
    b"<!DOCTYPE html",
    b"<!doctype html",
    b"<html",
    b"<HTML",
]

_READ_BYTES = 256  # Only need to peek at the start of the file


def is_html_file(path: Path) -> bool:
    """Return True if the file starts with an HTML signature.

    Parameters
    ----------
    path : Path
        Path to the file to inspect.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(_READ_BYTES)
        return any(header.lstrip().startswith(sig) for sig in _HTML_SIGNATURES)
    except OSError:
        return False


def is_empty_file(path: Path) -> bool:
    """Return True if the file is zero bytes."""
    try:
        return path.stat().st_size == 0
    except OSError:
        return False


def scan_video_dir(
    video_dir: Path,
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm"),
) -> tuple[list[Path], list[Path], list[Path]]:
    """Scan *video_dir* and classify each file.

    Returns
    -------
    valid : list[Path]
        Files that appear to be real binary video files.
    html : list[Path]
        Files whose content starts with an HTML signature.
    empty : list[Path]
        Zero-byte files.
    """
    valid: list[Path] = []
    html: list[Path] = []
    empty: list[Path] = []

    candidates = [p for p in video_dir.iterdir() if p.suffix.lower() in extensions]
    candidates.sort()

    for path in candidates:
        if is_empty_file(path):
            empty.append(path)
        elif is_html_file(path):
            html.append(path)
        else:
            valid.append(path)

    return valid, html, empty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and optionally remove HTML-disguised video files in WLASL raw_videos/"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing the downloaded .mp4 files",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help="Delete the invalid (HTML / empty) files after reporting",
    )
    parser.add_argument(
        "--save-valid",
        type=str,
        default=None,
        metavar="FILE",
        help="Write valid video IDs (stem names) to this text file, one per line",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        logger.error("Directory does not exist: %s", video_dir)
        sys.exit(1)

    logger.info("Scanning %s ...", video_dir)
    valid, html, empty = scan_video_dir(video_dir)

    total = len(valid) + len(html) + len(empty)

    print("\n" + "=" * 60)
    print("  WLASL Video Validation Report")
    print("=" * 60)
    print(f"  Total files scanned : {total}")
    print(f"  Valid video files   : {len(valid)}")
    print(f"  HTML redirect files : {len(html)}  ← fake videos (expired URLs)")
    print(f"  Empty files         : {len(empty)}")
    print()

    if html:
        print(f"  HTML files ({len(html)} total):")
        for p in html[:20]:
            print(f"    {p.name}")
        if len(html) > 20:
            print(f"    ... and {len(html) - 20} more")
        print()

    if empty:
        print(f"  Empty files ({len(empty)} total):")
        for p in empty[:20]:
            print(f"    {p.name}")
        if len(empty) > 20:
            print(f"    ... and {len(empty) - 20} more")
        print()

    if args.delete:
        bad = html + empty
        if not bad:
            print("  Nothing to delete.")
        else:
            print(f"  Deleting {len(bad)} invalid files...")
            for p in bad:
                try:
                    p.unlink()
                    logger.info("Deleted %s", p)
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", p, exc)
            print(f"  Done. {len(bad)} files removed.")
    else:
        if html or empty:
            print(
                "  Run with --delete to remove invalid files, e.g.:\n"
                f"    python scripts/validate_videos.py --video-dir {video_dir} --delete"
            )

    if args.save_valid:
        out_path = Path(args.save_valid)
        with open(out_path, "w", encoding="utf-8") as fh:
            for p in valid:
                fh.write(p.stem + "\n")
        print(f"\n  Valid video IDs written to: {out_path}  ({len(valid)} entries)")

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
