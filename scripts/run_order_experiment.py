#!/usr/bin/env python3
"""Plan and execute the JSON attribute-order experiment."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig

from logging_config import setup_logging
from src.main.order_experiment.io import (
    ensure_run_dirs,
    read_requests_parquet,
    read_samples_parquet,
    requests_path,
    samples_path,
    write_requests_parquet,
)
from src.main.order_experiment.planning import build_request_plan
from src.main.order_experiment.runner import run_experiment

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepare_plan(cfg: DictConfig) -> tuple[Path, Path]:
    run_id = str(cfg.RUN_ID)
    artifacts_dir = REPO_ROOT / cfg.ARTIFACTS_DIR
    ensure_run_dirs(artifacts_dir, run_id)
    req_path = requests_path(artifacts_dir, run_id)
    samp_path = samples_path(artifacts_dir, run_id)

    if req_path.exists() and bool(cfg.get("RESUME", True)):
        requests = read_requests_parquet(req_path)
        logger.info("reusing existing request plan (%d rows) at %s", len(requests), req_path)
        if requests.empty:
            requests = build_request_plan(cfg, REPO_ROOT)
            write_requests_parquet(req_path, requests)
            logger.info("wrote request plan (%d rows) to %s", len(requests), req_path)
    else:
        requests = build_request_plan(cfg, REPO_ROOT)
        write_requests_parquet(req_path, requests)
        logger.info("wrote request plan (%d rows) to %s", len(requests), req_path)

    n_stoch = int((~requests["is_greedy"]).sum())
    n_greedy = int(requests["is_greedy"].sum())
    logger.info(
        "plan summary mode=%s stochastic=%d greedy=%d total=%d",
        cfg.EXPERIMENT_MODE,
        n_stoch,
        n_greedy,
        len(requests),
    )
    return req_path, samp_path


async def _async_main(cfg: DictConfig) -> None:
    req_path, samp_path = _prepare_plan(cfg)
    requests = read_requests_parquet(req_path)
    samples = read_samples_parquet(samp_path)
    samples = await run_experiment(cfg, requests, samples, samp_path)
    logger.info("samples on disk: %d rows at %s", len(samples), samp_path)


@hydra.main(version_base=None, config_path="../conf", config_name="order_experiment_config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    asyncio.run(_async_main(cfg))


if __name__ == "__main__":
    main()
