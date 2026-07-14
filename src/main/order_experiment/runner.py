"""Chunked async executor for the attribute-order experiment."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.main.order_experiment.io import (
    append_samples_parquet,
    select_pending_requests,
)
from src.main.order_experiment.parse_order import annotate_sample_parse
from src.main.order_experiment.provider import (
    InfrastructureError,
    call_model_raw,
    sample_row_from_request,
)
from src.main.order_experiment.schemas import PARSE_STATUS_INFRA_ERROR

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request_optional_number(request: dict[str, Any], key: str) -> float | None:
    value = request.get(key, None)
    if value is None:
        return None
    try:
        if isinstance(value, float) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _request_top_k(request: dict[str, Any]) -> int | None:
    value = _request_optional_number(request, "top_k")
    return None if value is None else int(value)


def _request_top_p(request: dict[str, Any]) -> float | None:
    return _request_optional_number(request, "top_p")


async def _execute_one(
    request: dict[str, Any],
    *,
    semaphore: asyncio.Semaphore,
    retry_max: int,
    backoff_s: float,
) -> dict[str, Any]:
    attempts = 0
    last_error_class: str | None = None
    async with semaphore:
        while True:
            attempts += 1
            try:
                raw = await call_model_raw(
                    provider=str(request["provider"]),
                    modelo=str(request["modelo"]),
                    temperatura=float(request["temperatura"]),
                    system_prompt=str(request["system_text"]),
                    user_prompt=str(request["user_text"]),
                    top_k=_request_top_k(request),
                    top_p=_request_top_p(request),
                )
                parsed = annotate_sample_parse(
                    resposta_raw=raw,
                    requested_key_order=request["requested_key_order"],
                )
                return sample_row_from_request(
                    request,
                    resposta_raw=raw,
                    parse_status=str(parsed["parse_status"]),
                    observed_key_order=parsed["observed_key_order"],
                    order_match=parsed["order_match"],
                    error_class=None,
                    attempts=attempts,
                    completed_at=_utc_now(),
                )
            except InfrastructureError as exc:
                last_error_class = exc.error_class
                logger.warning(
                    "infra error request_id=%s attempt=%s/%s class=%s: %s",
                    request["request_id"],
                    attempts,
                    retry_max,
                    exc.error_class,
                    exc,
                )
                if attempts >= retry_max:
                    return sample_row_from_request(
                        request,
                        resposta_raw=None,
                        parse_status=PARSE_STATUS_INFRA_ERROR,
                        observed_key_order=None,
                        order_match=None,
                        error_class=last_error_class,
                        attempts=attempts,
                        completed_at=_utc_now(),
                    )
                await asyncio.sleep(backoff_s * attempts)


async def run_experiment(
    cfg: DictConfig,
    requests: pd.DataFrame,
    samples: pd.DataFrame,
    samples_file,
) -> pd.DataFrame:
    """Execute pending requests with bounded concurrency; flush to Parquet."""
    retry_infra = bool(cfg.get("RETRY_INFRA_ON_RESUME", True))
    pending = select_pending_requests(
        requests, samples, retry_infra=retry_infra and bool(cfg.get("RESUME", True))
    )
    total_pending = len(pending)
    logger.info(
        "pending requests: %d / %d (resume=%s retry_infra=%s)",
        total_pending,
        len(requests),
        cfg.get("RESUME", True),
        retry_infra,
    )
    if total_pending == 0:
        return samples

    max_in_flight = max(1, int(cfg.get("MAX_IN_FLIGHT", 32)))
    chunk_size = max(1, int(cfg.get("SCHEDULE_CHUNK_SIZE", 256)))
    flush_every = max(1, int(cfg.get("FLUSH_EVERY", 64)))
    retry_max = max(1, int(cfg.get("RETRY_MAX_ATTEMPTS", 3)))
    backoff_s = float(cfg.get("RETRY_BACKOFF_S", 1.0))

    semaphore = asyncio.Semaphore(max_in_flight)
    buffer: list[dict[str, Any]] = []
    completed = 0

    records = pending.to_dict(orient="records")
    with tqdm(total=total_pending, desc="order-exp", unit="req") as pbar:
        for start in range(0, total_pending, chunk_size):
            chunk = records[start : start + chunk_size]
            # Cap concurrent tasks to chunk size, but semaphore enforces MAX_IN_FLIGHT.
            tasks = [
                asyncio.create_task(
                    _execute_one(
                        row,
                        semaphore=semaphore,
                        retry_max=retry_max,
                        backoff_s=backoff_s,
                    )
                )
                for row in chunk
            ]
            for coro in asyncio.as_completed(tasks):
                row = await coro
                buffer.append(row)
                completed += 1
                pbar.update(1)
                if len(buffer) >= flush_every:
                    samples = append_samples_parquet(samples_file, buffer)
                    logger.info(
                        "flushed %d samples (completed %d / %d)",
                        len(buffer),
                        completed,
                        total_pending,
                    )
                    buffer.clear()

        if buffer:
            samples = append_samples_parquet(samples_file, buffer)
            logger.info("final flush of %d samples", len(buffer))
            buffer.clear()

    return samples
