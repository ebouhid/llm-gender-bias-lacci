"""Parquet I/O and verbalization subset filters for NLA artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.main.nla.schemas import (
    ACTIVATION_COLUMNS,
    RECONSTRUCTION_COLUMNS,
    VERBALIZATION_COLUMNS,
)


def artifact_path(artifacts_dir: str | Path, kind: str, run_id: str) -> Path:
    return Path(artifacts_dir) / kind / f"{run_id}.parquet"


def ensure_artifact_dirs(artifacts_dir: str | Path) -> None:
    root = Path(artifacts_dir)
    for sub in ("activations", "verbalizations", "reconstructions", "analysis", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def write_activations_parquet(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no activation rows to write")

    df = pd.DataFrame(rows)
    for col in ACTIVATION_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"missing activation column: {col}")

    vectors = np.stack(
        [np.asarray(v, dtype=np.float32) for v in df["activation_vector"]]
    )
    d_model = vectors.shape[1]
    flat = pa.array(vectors.reshape(-1), type=pa.float32())
    vector_col = pa.FixedSizeListArray.from_arrays(flat, d_model)

    table = pa.table(
        {
            "activation_id": df["activation_id"].astype(str).tolist(),
            "run_id": df["run_id"].astype(str).tolist(),
            "example_id": df["example_id"].astype(str).tolist(),
            "condition_id": df["condition_id"].astype(str).tolist(),
            "modelo": df["modelo"].astype(str).tolist(),
            "temperatura": df["temperatura"].astype(float).tolist(),
            "repeticao": df["repeticao"].astype(int).tolist(),
            "marcador_codigo": df["marcador_codigo"].astype(str).tolist(),
            "disciplina_codigo": df["disciplina_codigo"].astype(str).tolist(),
            "prompt_hash": df["prompt_hash"].astype(str).tolist(),
            "base_model": df["base_model"].astype(str).tolist(),
            "nla_layer": df["nla_layer"].astype(int).tolist(),
            "token_index": df["token_index"].astype(int).tolist(),
            "token_id": df["token_id"].astype(int).tolist(),
            "token_str": df["token_str"].astype(str).tolist(),
            "token_role": df["token_role"].astype(str).tolist(),
            "char_start": df["char_start"].astype(int).tolist(),
            "char_end": df["char_end"].astype(int).tolist(),
            "activation_vector": vector_col,
        }
    )
    pq.write_table(table, path)


def read_activations_parquet(path: str | Path) -> pd.DataFrame:
    table = pq.read_table(path)
    df = table.to_pandas()

    if "activation_vector" in df.columns:
        stacked = np.stack(
            [np.asarray(v, dtype=np.float32) for v in df["activation_vector"]]
        )
        df["activation_vector"] = list(stacked)

    return df


def write_verbalizations_parquet(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in VERBALIZATION_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[list(VERBALIZATION_COLUMNS)]
    df.to_parquet(path, index=False)


def read_verbalizations_parquet(path: str | Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=list(VERBALIZATION_COLUMNS))
    return pd.read_parquet(path)


def append_verbalizations_parquet(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    existing = read_verbalizations_parquet(path)
    new_df = pd.DataFrame(rows)
    for col in VERBALIZATION_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[list(VERBALIZATION_COLUMNS)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["activation_id"], keep="last")
    write_verbalizations_parquet(path, combined.to_dict(orient="records"))


def write_reconstructions_parquet(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in RECONSTRUCTION_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"missing reconstruction column: {col}")
    df = df[list(RECONSTRUCTION_COLUMNS)]
    df.to_parquet(path, index=False)


def read_reconstructions_parquet(path: str | Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=list(RECONSTRUCTION_COLUMNS))
    return pd.read_parquet(path)


def append_reconstructions_parquet(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    existing = read_reconstructions_parquet(path)
    new_df = pd.DataFrame(rows)
    for col in RECONSTRUCTION_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[list(RECONSTRUCTION_COLUMNS)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["activation_id"], keep="last")
    write_reconstructions_parquet(path, combined.to_dict(orient="records"))


def select_verbalization_subset(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    if tier == "tier4":
        return df.copy()

    if tier == "tier3":
        return df[df["token_role"] != "generated_output"].copy()

    if tier == "tier2":
        user_roles = {"user_prompt", "task_instruction", "answer_space"}
        return df[df["token_role"].isin(user_roles)].copy()

    if tier == "tier1":
        selected_idx: list[int] = []
        for _, group in df.groupby("example_id"):
            marker_positions: set[int] = set()
            markers = group[group["token_role"] == "demographic_marker"]["token_index"]
            for idx in markers:
                lo = max(0, int(idx) - 5)
                hi = int(idx) + 5
                marker_positions.update(range(lo, hi + 1))
            final_idx = int(group["token_index"].max())
            marker_positions.add(final_idx)
            selected_idx.extend(
                group[group["token_index"].isin(marker_positions)].index.tolist()
            )
        return df.loc[selected_idx].copy()

    raise ValueError(f"unknown verbalization tier: {tier!r}")
