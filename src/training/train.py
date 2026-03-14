"""
Training entry point for WLASL sign language recognition.

Dispatches to the prototypical (episodic) training loop using the
ST-GCN encoder with metric learning.
"""

import argparse
import logging

from src.training.config import load_config
from src.training.train_prototypical import main

logger = logging.getLogger(__name__)


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
