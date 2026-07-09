#!/usr/bin/env python3
"""Run NLA verbalization over saved activation rows."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.main.nla.ids import make_run_id
from src.main.nla.io import (
    append_verbalizations_parquet,
    artifact_path,
    ensure_artifact_dirs,
    read_activations_parquet,
    read_verbalizations_parquet,
    select_verbalization_subset,
)
from src.main.nla.nla_client import NLAClient

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../conf", config_name="nla_config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_id = make_run_id(cfg.get("RUN_ID"))
    tier = cfg.get("VERBALIZATION_TIER", "tier1")
    resume = bool(cfg.get("RESUME", True))
    save_raw = bool(cfg.get("SAVE_RAW", False))
    artifacts_dir = REPO_ROOT / cfg.ARTIFACTS_DIR
    ensure_artifact_dirs(artifacts_dir)

    activations_path = artifact_path(artifacts_dir, "activations", run_id)
    verbalizations_path = artifact_path(artifacts_dir, "verbalizations", run_id)

    df = read_activations_parquet(activations_path)
    subset = select_verbalization_subset(df, tier)
    logger.info(
        "verbalizing tier=%s: %d / %d rows from %s",
        tier,
        len(subset),
        len(df),
        activations_path,
    )

    if resume:
        done = set(read_verbalizations_parquet(verbalizations_path)["activation_id"].tolist())
        subset = subset[~subset["activation_id"].isin(done)]
        logger.info("resume enabled: %d rows remaining", len(subset))

    if subset.empty:
        logger.info("nothing to verbalize")
        return

    client = NLAClient(
        cfg.AV_CHECKPOINT,
        sglang_url=cfg.SGLANG_NLA_URL,
    )

    rows: list[dict] = []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="verbalize"):
        vector = row["activation_vector"]
        if save_raw:
            explanation, raw = client.generate_with_raw(
                vector,
                temperature=float(cfg.TEMPERATURE),
                max_new_tokens=int(cfg.MAX_NEW_TOKENS),
            )
        else:
            explanation = client.generate(
                vector,
                temperature=float(cfg.TEMPERATURE),
                max_new_tokens=int(cfg.MAX_NEW_TOKENS),
            )
            raw = None

        rows.append(
            {
                "activation_id": row["activation_id"],
                "nla_av_checkpoint": str(cfg.AV_CHECKPOINT),
                "nla_explanation": explanation,
                "nla_raw_response": raw,
                "temperature": float(cfg.TEMPERATURE),
                "max_new_tokens": int(cfg.MAX_NEW_TOKENS),
                "verbalized_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        if len(rows) >= 10:
            append_verbalizations_parquet(verbalizations_path, rows)
            rows = []

    if rows:
        append_verbalizations_parquet(verbalizations_path, rows)

    logger.info("saved verbalizations to %s", verbalizations_path)


if __name__ == "__main__":
    main()
