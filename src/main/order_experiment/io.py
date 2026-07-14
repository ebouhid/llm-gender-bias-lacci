"""Parquet I/O for order-experiment request and sample artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.main.order_experiment.schemas import (
    PARSE_STATUS_INFRA_ERROR,
    REQUEST_COLUMNS,
    SAMPLE_COLUMNS,
)


def run_dir(artifacts_dir: str | Path, run_id: str) -> Path:
    return Path(artifacts_dir) / run_id


def ensure_run_dirs(artifacts_dir: str | Path, run_id: str) -> Path:
    root = run_dir(artifacts_dir, run_id)
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    return root


def requests_path(artifacts_dir: str | Path, run_id: str) -> Path:
    return run_dir(artifacts_dir, run_id) / "requests.parquet"


def samples_path(artifacts_dir: str | Path, run_id: str) -> Path:
    return run_dir(artifacts_dir, run_id) / "samples.parquet"


def analysis_dir(artifacts_dir: str | Path, run_id: str) -> Path:
    return run_dir(artifacts_dir, run_id) / "analysis"


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_requests_parquet(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    out = df.reindex(columns=list(REQUEST_COLUMNS))
    _atomic_write_parquet(out, path)


def read_requests_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(REQUEST_COLUMNS))
    df = pd.read_parquet(path)
    return df.reindex(columns=list(REQUEST_COLUMNS))


def write_samples_parquet(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    out = df.reindex(columns=list(SAMPLE_COLUMNS))
    _atomic_write_parquet(out, path)


def read_samples_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(SAMPLE_COLUMNS))
    df = pd.read_parquet(path)
    return df.reindex(columns=list(SAMPLE_COLUMNS))


def append_samples_parquet(path: str | Path, rows: list[dict]) -> pd.DataFrame:
    """Append sample rows and dedupe by request_id (keep last)."""
    if not rows:
        return read_samples_parquet(path)
    new_df = pd.DataFrame(rows)
    for col in SAMPLE_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df.reindex(columns=list(SAMPLE_COLUMNS))
    path = Path(path)
    if not path.exists():
        write_samples_parquet(path, new_df)
        return new_df
    existing = read_samples_parquet(path)
    if existing.empty:
        write_samples_parquet(path, new_df)
        return new_df
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["request_id"], keep="last")
    write_samples_parquet(path, combined)
    return combined


def completed_request_ids(
    samples: pd.DataFrame,
    *,
    retry_infra: bool = True,
) -> set[str]:
    """IDs that should not be re-executed.

    Successful and malformed/empty rows are immutable. Infra-error rows are
    treated as incomplete when retry_infra is True.
    """
    if samples.empty:
        return set()
    if not retry_infra:
        return set(samples["request_id"].astype(str))
    keep = samples["parse_status"].astype(str) != PARSE_STATUS_INFRA_ERROR
    return set(samples.loc[keep, "request_id"].astype(str))


def select_pending_requests(
    requests: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    retry_infra: bool = True,
) -> pd.DataFrame:
    done = completed_request_ids(samples, retry_infra=retry_infra)
    if not done:
        return requests.copy()
    mask = ~requests["request_id"].astype(str).isin(done)
    return requests.loc[mask].copy()
