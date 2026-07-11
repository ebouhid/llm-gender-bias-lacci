#!/usr/bin/env python3
"""Run NLA verbalization over saved activation rows."""

from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _verbalize_chunk(
    client: NLAClient,
    chunk_ids: list[str],
    chunk_vecs: list,
    *,
    save_raw: bool,
    av_checkpoint: str,
    temperature: float,
    max_new_tokens: int,
    sampling: dict,
) -> list[dict]:
    verbalized_at = datetime.now(timezone.utc).isoformat()
    if save_raw:
        results = client.generate_batch_with_raw(chunk_vecs, **sampling)
        return [
            {
                "activation_id": activation_id,
                "nla_av_checkpoint": av_checkpoint,
                "nla_explanation": explanation,
                "nla_raw_response": raw,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "verbalized_at": verbalized_at,
            }
            for activation_id, (explanation, raw) in zip(chunk_ids, results)
        ]

    explanations = client.generate_batch(chunk_vecs, **sampling)
    return [
        {
            "activation_id": activation_id,
            "nla_av_checkpoint": av_checkpoint,
            "nla_explanation": explanation,
            "nla_raw_response": None,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "verbalized_at": verbalized_at,
        }
        for activation_id, explanation in zip(chunk_ids, explanations)
    ]


@hydra.main(version_base=None, config_path="../conf", config_name="nla_config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_id = make_run_id(cfg.get("RUN_ID"))
    tier = cfg.get("VERBALIZATION_TIER", "tier1")
    resume = bool(cfg.get("RESUME", True))
    save_raw = bool(cfg.get("SAVE_RAW", False))
    batch_size = max(1, int(cfg.get("VERBALIZATION_BATCH_SIZE", 8)))
    concurrency = max(1, int(cfg.get("VERBALIZATION_CONCURRENCY", 1)))
    flush_rows = max(batch_size, int(cfg.get("VERBALIZATION_FLUSH_ROWS", 512)))
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

    activation_ids = subset["activation_id"].tolist()
    vectors = subset["activation_vector"].tolist()
    sampling = {
        "temperature": float(cfg.TEMPERATURE),
        "max_new_tokens": int(cfg.MAX_NEW_TOKENS),
    }
    av_checkpoint = str(cfg.AV_CHECKPOINT)
    temperature = float(cfg.TEMPERATURE)
    max_new_tokens = int(cfg.MAX_NEW_TOKENS)

    chunks = [
        (activation_ids[start : start + batch_size], vectors[start : start + batch_size])
        for start in range(0, len(vectors), batch_size)
    ]
    logger.info(
        "verbalization batch_size=%d concurrency=%d flush_rows=%d chunks=%d",
        batch_size,
        concurrency,
        flush_rows,
        len(chunks),
    )

    thread_local = threading.local()

    def get_client() -> NLAClient:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = NLAClient(
                cfg.AV_CHECKPOINT,
                sglang_url=cfg.SGLANG_NLA_URL,
            )
            thread_local.client = client
        return client

    def process_chunk(chunk: tuple[list[str], list]) -> list[dict]:
        chunk_ids, chunk_vecs = chunk
        t0 = time.perf_counter()
        rows = _verbalize_chunk(
            get_client(),
            chunk_ids,
            chunk_vecs,
            save_raw=save_raw,
            av_checkpoint=av_checkpoint,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
        )
        elapsed = time.perf_counter() - t0
        logger.debug(
            "chunk size=%d elapsed=%.2fs (%.2fs/item)",
            len(chunk_ids),
            elapsed,
            elapsed / len(chunk_ids),
        )
        return rows

    buffer: list[dict] = []
    completed_items = 0
    t_start = time.perf_counter()

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            append_verbalizations_parquet(verbalizations_path, buffer)
            buffer = []

    if concurrency <= 1:
        for chunk in tqdm(chunks, desc="verbalize"):
            buffer.extend(process_chunk(chunk))
            completed_items += len(chunk[0])
            if len(buffer) >= flush_rows:
                flush_buffer()
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(process_chunk, chunk): chunk for chunk in chunks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="verbalize"):
                rows = future.result()
                buffer.extend(rows)
                completed_items += len(rows)
                if len(buffer) >= flush_rows:
                    flush_buffer()

    flush_buffer()

    elapsed = time.perf_counter() - t_start
    if completed_items:
        logger.info(
            "verbalized %d items in %.1fs (%.2fs/item)",
            completed_items,
            elapsed,
            elapsed / completed_items,
        )
    logger.info("saved verbalizations to %s", verbalizations_path)


if __name__ == "__main__":
    main()
