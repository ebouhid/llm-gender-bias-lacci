#!/usr/bin/env python3
"""Produce minimal CSV summaries from NLA gender-bias artifacts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig

from src.main.nla.analysis import (
    build_joined_table,
    outcome_linked_summary,
    span_level_summary,
    token_local_contrast,
)
from src.main.nla.ids import make_run_id
from src.main.nla.io import (
    artifact_path,
    ensure_artifact_dirs,
    read_activations_parquet,
    read_reconstructions_parquet,
    read_verbalizations_parquet,
)

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../conf", config_name="nla_config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_id = make_run_id(cfg.get("RUN_ID"))
    artifacts_dir = REPO_ROOT / cfg.ARTIFACTS_DIR
    ensure_artifact_dirs(artifacts_dir)

    activations = read_activations_parquet(
        artifact_path(artifacts_dir, "activations", run_id)
    )
    verbalizations = read_verbalizations_parquet(
        artifact_path(artifacts_dir, "verbalizations", run_id)
    )
    reconstructions = read_reconstructions_parquet(
        artifact_path(artifacts_dir, "reconstructions", run_id)
    )

    joined = build_joined_table(
        activations,
        verbalizations,
        reconstructions,
        generation_jsonl=REPO_ROOT / cfg.get("GENERATION_JSONL"),
    )
    logger.info("joined table has %d rows", len(joined))

    out_dir = Path(artifacts_dir) / "analysis" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    token_local = token_local_contrast(joined)
    span_level = span_level_summary(joined)
    outcome_linked = outcome_linked_summary(joined)

    token_local.to_csv(out_dir / "token_local_contrast.csv", index=False)
    span_level.to_csv(out_dir / "span_level_summary.csv", index=False)
    outcome_linked.to_csv(out_dir / "outcome_linked_summary.csv", index=False)

    logger.info("saved analysis CSVs to %s", out_dir)


if __name__ == "__main__":
    main()
