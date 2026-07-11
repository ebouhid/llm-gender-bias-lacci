#!/usr/bin/env python3
"""Score NLA reconstruction errors for verbalized activations."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.main.nla.activation_extraction import _resolve_dtype
from src.main.nla.ids import make_run_id
from src.main.nla.io import (
    append_reconstructions_parquet,
    artifact_path,
    ensure_artifact_dirs,
    read_activations_parquet,
    read_reconstructions_parquet,
    read_verbalizations_parquet,
)
from src.main.nla.reconstruction import NLACritic

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../conf", config_name="nla_config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_id = make_run_id(cfg.get("RUN_ID"))
    resume = bool(cfg.get("RESUME", True))
    batch_size = max(1, int(cfg.get("RECONSTRUCTION_BATCH_SIZE", 32)))
    flush_rows = max(batch_size, int(cfg.get("RECONSTRUCTION_FLUSH_ROWS", 512)))
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

    critic = NLACritic(
        cfg.AR_CHECKPOINT,
        device=cfg.DEVICE,
        dtype=_resolve_dtype(cfg.DTYPE),
    )

    activation_ids = joined["activation_id"].tolist()
    explanations = joined["nla_explanation"].tolist()
    vectors = joined["activation_vector"].tolist()
    ar_checkpoint = str(cfg.AR_CHECKPOINT)

    chunks = [
        (
            activation_ids[start : start + batch_size],
            explanations[start : start + batch_size],
            vectors[start : start + batch_size],
        )
        for start in range(0, len(activation_ids), batch_size)
    ]
    logger.info(
        "reconstruction batch_size=%d flush_rows=%d chunks=%d",
        batch_size,
        flush_rows,
        len(chunks),
    )

    buffer: list[dict] = []
    completed_items = 0
    t_start = time.perf_counter()

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            append_reconstructions_parquet(recon_path, buffer)
            buffer = []

    for chunk_ids, chunk_explanations, chunk_vectors in tqdm(
        chunks, desc="reconstruct"
    ):
        scores = critic.score_batch_with_norms(chunk_explanations, chunk_vectors)
        buffer.extend(
            {
                "activation_id": activation_id,
                "nla_ar_checkpoint": ar_checkpoint,
                "reconstruction_mse": mse,
                "reconstruction_cosine": cos,
                "original_norm": original_norm,
                "reconstructed_norm": reconstructed_norm,
            }
            for activation_id, (mse, cos, original_norm, reconstructed_norm) in zip(
                chunk_ids, scores
            )
        )
        completed_items += len(chunk_ids)
        if len(buffer) >= flush_rows:
            flush_buffer()

    flush_buffer()

    elapsed = time.perf_counter() - t_start
    if completed_items:
        logger.info(
            "reconstructed %d items in %.1fs (%.2fs/item)",
            completed_items,
            elapsed,
            elapsed / completed_items,
        )
    logger.info("saved reconstructions to %s", recon_path)


if __name__ == "__main__":
    main()
