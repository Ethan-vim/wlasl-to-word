"""
Diagnostic script to verify MediaPipe installation.

Checks whether mediapipe is installed correctly and the 'solutions' module
(specifically Holistic) is accessible.  Prints version, platform, and
available modules to help troubleshoot installation issues.

Usage:
    python scripts/check_mediapipe.py
"""

import platform
import sys


def main() -> None:
    print("=" * 60)
    print("  MediaPipe Installation Check")
    print("=" * 60)
    print()
    print(f"  Python      : {sys.version}")
    print(f"  Platform    : {platform.platform()}")
    print(f"  Arch        : {platform.machine()}")
    print()

    # Check if mediapipe is installed at all
    try:
        import mediapipe as mp
    except ImportError:
        print("  [FAIL] mediapipe is NOT installed.")
        print()
        print("  Install it with:")
        print("    pip install mediapipe==0.10.11")
        print()
        print("  On Apple Silicon Mac:")
        print("    pip install mediapipe-silicon")
        sys.exit(1)

    version = getattr(mp, "__version__", "unknown")
    print(f"  mediapipe   : {version}")

    # List top-level attributes
    attrs = [x for x in dir(mp) if not x.startswith("_")]
    print(f"  Top-level   : {attrs}")
    print()

    # Check standard path: mp.solutions
    has_solutions = hasattr(mp, "solutions")
    if has_solutions:
        print("  [OK]   mp.solutions is available (standard path)")
    else:
        print("  [WARN] mp.solutions is NOT available")

    # Check internal path: mediapipe.python.solutions
    has_internal = False
    try:
        from mediapipe.python.solutions import holistic  # noqa: F401

        has_internal = True
        print("  [OK]   mediapipe.python.solutions.holistic is available (fallback path)")
    except (ImportError, ModuleNotFoundError):
        print("  [WARN] mediapipe.python.solutions.holistic is NOT available")

    print()

    # Check Holistic specifically
    holistic_mod = None
    if has_solutions:
        try:
            holistic_mod = mp.solutions.holistic
            print("  [OK]   mp.solutions.holistic.Holistic class found")
        except AttributeError:
            print("  [FAIL] mp.solutions exists but holistic is missing")
    elif has_internal:
        try:
            from mediapipe.python.solutions import holistic as holistic_mod

            print("  [OK]   Holistic class found via fallback path")
        except (ImportError, ModuleNotFoundError):
            pass

    print()

    if holistic_mod is not None:
        print("  RESULT: MediaPipe is working correctly.")
        print("  The preprocessing pipeline will use the available import path.")
        print()
        sys.exit(0)
    else:
        print("  RESULT: MediaPipe Holistic is NOT accessible.")
        print()
        print("  Recommended fix:")
        print(f"    pip install --force-reinstall mediapipe==0.10.11")
        print()
        print("  If that fails, try:")
        print(f"    pip install --force-reinstall mediapipe==0.10.9")
        print()
        if platform.machine() in ("arm64", "aarch64"):
            print("  On Apple Silicon Mac:")
            print("    pip install mediapipe-silicon")
            print()
        sys.exit(1)


if __name__ == "__main__":
    main()
