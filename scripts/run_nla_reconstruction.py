#!/usr/bin/env python3
"""Score NLA reconstruction errors for verbalized activations."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

from src.main.nla.ids import make_run_id
from src.main.nla.io import (
    artifact_path,
    ensure_artifact_dirs,
    read_activations_parquet,
    read_reconstructions_parquet,
    read_verbalizations_parquet,
    write_reconstructions_parquet,
)
from src.main.nla.reconstruction import NLACritic

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../conf", config_name="nla_config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_id = make_run_id(cfg.get("RUN_ID"))
    resume = bool(cfg.get("RESUME", True))
    artifacts_dir = REPO_ROOT / cfg.ARTIFACTS_DIR
    ensure_artifact_dirs(artifacts_dir)

    activations = read_activations_parquet(
        artifact_path(artifacts_dir, "activations", run_id)
    )
    verbalizations = read_verbalizations_parquet(
        artifact_path(artifacts_dir, "verbalizations", run_id)
    )
    recon_path = artifact_path(artifacts_dir, "reconstructions", run_id)

    joined = verbalizations.merge(
        activations[
            [
                "activation_id",
                "activation_vector",
            ]
        ],
        on="activation_id",
        how="inner",
    )
    logger.info("scoring %d verbalized activations", len(joined))

    if resume:
        done = set(read_reconstructions_parquet(recon_path)["activation_id"].tolist())
        joined = joined[~joined["activation_id"].isin(done)]
        logger.info("resume enabled: %d rows remaining", len(joined))

    if joined.empty:
        logger.info("nothing to reconstruct")
        return

    critic = NLACritic(cfg.AR_CHECKPOINT, device=cfg.DEVICE)

    rows: list[dict] = []
    for _, row in tqdm(joined.iterrows(), total=len(joined), desc="reconstruct"):
        mse, cos, original_norm, reconstructed_norm = critic.score_with_norms(
            row["nla_explanation"],
            row["activation_vector"],
        )
        rows.append(
            {
                "activation_id": row["activation_id"],
                "nla_ar_checkpoint": str(cfg.AR_CHECKPOINT),
                "reconstruction_mse": mse,
                "reconstruction_cosine": cos,
                "original_norm": original_norm,
                "reconstructed_norm": reconstructed_norm,
            }
        )

    if resume and recon_path.exists():
        existing = read_reconstructions_parquet(recon_path)
        combined = pd.concat(
            [existing, pd.DataFrame(rows)], ignore_index=True
        ).drop_duplicates(subset=["activation_id"], keep="last")
        rows = combined.to_dict(orient="records")

    write_reconstructions_parquet(recon_path, rows)
    logger.info("saved reconstructions to %s", recon_path)


if __name__ == "__main__":
    main()
