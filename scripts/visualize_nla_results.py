#!/usr/bin/env python3
"""Build static HTML reports and figures from NLA merged analysis results."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig

from src.main.nla.analysis import (
    build_joined_table,
    read_merged_results,
    write_merged_results,
)
from src.main.nla.ids import make_run_id
from src.main.nla.io import (
    artifact_path,
    ensure_artifact_dirs,
    read_activations_parquet,
    read_reconstructions_parquet,
    read_verbalizations_parquet,
)
from src.main.nla.report import build_html_report

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_run_id(cfg: DictConfig, artifacts_dir: Path) -> str:
    """Prefer explicit RUN_ID; otherwise pick the newest complete artifact trio."""
    explicit = cfg.get("RUN_ID")
    if explicit:
        return make_run_id(explicit)

    act_dir = artifacts_dir / "activations"
    if not act_dir.exists():
        raise FileNotFoundError(f"no activations under {act_dir}")

    candidates = sorted(act_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        run_id = path.stem
        verb = artifact_path(artifacts_dir, "verbalizations", run_id)
        recon = artifact_path(artifacts_dir, "reconstructions", run_id)
        if verb.exists() and recon.exists():
            return run_id

    raise FileNotFoundError(
        f"no run_id with activations+verbalizations+reconstructions under {artifacts_dir}"
    )


def load_or_build_merged(
    artifacts_dir: Path,
    run_id: str,
    generation_jsonl: Path | None,
) -> Path:
    """Load merged_results.parquet, rebuilding from stage artifacts if needed."""
    analysis_dir = artifacts_dir / "analysis" / run_id
    merged_path = analysis_dir / "merged_results.parquet"
    if merged_path.exists():
        logger.info("loading existing merged table %s", merged_path)
        return merged_path

    logger.info("merged_results missing; rebuilding join for run_id=%s", run_id)
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
        generation_jsonl=generation_jsonl,
    )
    write_merged_results(merged_path, joined)
    logger.info("wrote %s (%d rows)", merged_path, len(joined))
    return merged_path


@hydra.main(version_base=None, config_path="../conf", config_name="nla_config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    artifacts_dir = REPO_ROOT / cfg.ARTIFACTS_DIR
    ensure_artifact_dirs(artifacts_dir)

    run_id = resolve_run_id(cfg, artifacts_dir)
    logger.info("visualizing run_id=%s", run_id)

    gen_path = REPO_ROOT / cfg.get("GENERATION_JSONL") if cfg.get("GENERATION_JSONL") else None
    merged_path = load_or_build_merged(artifacts_dir, run_id, gen_path)
    merged = read_merged_results(merged_path)

    out_dir = artifacts_dir / "reports" / run_id
    max_samples = cfg.get("REPORT_MAX_SAMPLES")
    max_table_rows = int(cfg.get("REPORT_MAX_TABLE_ROWS", 2000))

    index_path = build_html_report(
        merged,
        out_dir=out_dir,
        run_id=run_id,
        max_table_rows=max_table_rows,
        max_sample_pages=int(max_samples) if max_samples is not None else None,
    )
    logger.info("report ready: %s", index_path)


if __name__ == "__main__":
    main()
