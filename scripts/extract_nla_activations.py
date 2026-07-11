#!/usr/bin/env python3
"""Extract per-token residual activations for the NLA pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig

from src.main.nla.activation_extraction import extract_all_activations
from src.main.nla.ids import make_run_id
from src.main.nla.io import artifact_path, ensure_artifact_dirs, write_activations_parquet

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../conf", config_name="nla_config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_id = make_run_id(cfg.get("RUN_ID"))
    artifacts_dir = REPO_ROOT / cfg.ARTIFACTS_DIR
    ensure_artifact_dirs(artifacts_dir)

    logger.info("run_id=%s base_model=%s layer=%s", run_id, cfg.BASE_MODEL, cfg.NLA_LAYER)
    gen_path = REPO_ROOT / cfg.GENERATION_JSONL if cfg.get("GENERATION_JSONL") else None
    rows = extract_all_activations(
        cfg=cfg,
        run_id=run_id,
        base_model=cfg.BASE_MODEL,
        nla_layer=int(cfg.NLA_LAYER),
        device=cfg.DEVICE,
        dtype_name=cfg.DTYPE,
        generation_jsonl=gen_path,
    )
    logger.info("extracted %d token activation rows", len(rows))

    out_path = artifact_path(artifacts_dir, "activations", run_id)
    write_activations_parquet(out_path, rows)
    logger.info("saved activations to %s", out_path)


if __name__ == "__main__":
    main()
