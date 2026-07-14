#!/usr/bin/env python3
"""Analyze JSON attribute-order experiment samples."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig

from logging_config import setup_logging
from src.main.order_experiment.analysis import run_analysis

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../conf", config_name="order_experiment_config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    artifacts_dir = REPO_ROOT / cfg.ARTIFACTS_DIR
    run_id = str(cfg.RUN_ID)
    paths = run_analysis(artifacts_dir=artifacts_dir, run_id=run_id)
    for name, path in paths.items():
        logger.info("wrote %s -> %s", name, path)


if __name__ == "__main__":
    main()
