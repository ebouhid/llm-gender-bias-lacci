"""Summarize demographic distributions and order compliance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.main.order_experiment.io import analysis_dir, read_samples_parquet, samples_path
from src.main.order_experiment.parse_order import extract_json_object_span
from src.main.order_experiment.schemas import PARSE_STATUS_OK

_DEMOGRAPHIC_KEYS = (
    "sexo_atribuido",
    "cor_ou_raca",
    "estado",
    "idade",
    "renda_mensal",
)


def _parse_obj(raw: Any) -> dict[str, Any] | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw)
    blob = extract_json_object_span(text)
    if blob is None:
        return None
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def enrich_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Add extracted demographic columns from resposta_raw."""
    if df.empty:
        cols = list(df.columns) + list(_DEMOGRAPHIC_KEYS)
        return pd.DataFrame(columns=cols)

    records = []
    for _, row in df.iterrows():
        obj = _parse_obj(row.get("resposta_raw"))
        extracted = {k: (obj.get(k) if obj else None) for k in _DEMOGRAPHIC_KEYS}
        records.append(extracted)
    extra = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), extra], axis=1)


def order_compliance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of order_match by order_id × temperatura (and greedy flag)."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "order_id",
                "temperatura",
                "is_greedy",
                "n",
                "n_ok_parse",
                "n_order_match",
                "order_match_rate",
            ]
        )
    work = df.copy()
    work["order_match_bool"] = work["order_match"].map(
        lambda x: True if x is True or x == True else (False if x is False or x == False else None)
    )
    rows = []
    for keys, grp in work.groupby(["order_id", "temperatura", "is_greedy"], dropna=False):
        order_id, temperatura, is_greedy = keys
        n = len(grp)
        n_ok = int((grp["parse_status"] == PARSE_STATUS_OK).sum())
        matched = grp["order_match_bool"].dropna()
        n_match = int(matched.sum()) if len(matched) else 0
        n_compared = len(matched)
        rows.append(
            {
                "order_id": order_id,
                "temperatura": float(temperatura),
                "is_greedy": bool(is_greedy),
                "n": n,
                "n_ok_parse": n_ok,
                "n_order_match": n_match,
                "n_order_compared": n_compared,
                "order_match_rate": (n_match / n_compared) if n_compared else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["is_greedy", "temperatura", "order_id"])


def distribution_table(
    df: pd.DataFrame,
    column: str,
    *,
    stochastic_only: bool = True,
) -> pd.DataFrame:
    """Frequency and proportion of a demographic column by order × temperature."""
    work = df.copy()
    if stochastic_only and "is_greedy" in work.columns:
        work = work.loc[~work["is_greedy"].astype(bool)]
    work = work.loc[work["parse_status"] == PARSE_STATUS_OK]
    if work.empty or column not in work.columns:
        return pd.DataFrame(
            columns=["order_id", "temperatura", column, "count", "proportion"]
        )
    grouped = (
        work.groupby(["order_id", "temperatura", column], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = (
        work.groupby(["order_id", "temperatura"], dropna=False)
        .size()
        .reset_index(name="total")
    )
    out = grouped.merge(totals, on=["order_id", "temperatura"], how="left")
    out["proportion"] = out["count"] / out["total"]
    return out.sort_values(["temperatura", "order_id", column])


def parse_status_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["parse_status", "is_greedy", "count"])
    return (
        df.groupby(["parse_status", "is_greedy"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["is_greedy", "parse_status"])
    )


def run_analysis(
    *,
    artifacts_dir: str | Path,
    run_id: str,
) -> dict[str, Path]:
    """Write analysis CSVs under artifacts/order_experiment/{run_id}/analysis/."""
    samp = read_samples_parquet(samples_path(artifacts_dir, run_id))
    enriched = enrich_samples(samp)
    out_dir = analysis_dir(artifacts_dir, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    compliance = order_compliance_table(enriched)
    path = out_dir / "order_compliance.csv"
    compliance.to_csv(path, index=False)
    paths["order_compliance"] = path

    status = parse_status_table(enriched)
    path = out_dir / "parse_status.csv"
    status.to_csv(path, index=False)
    paths["parse_status"] = path

    for col in ("sexo_atribuido", "cor_ou_raca", "estado"):
        dist = distribution_table(enriched, col, stochastic_only=True)
        path = out_dir / f"dist_{col}_stochastic.csv"
        dist.to_csv(path, index=False)
        paths[f"dist_{col}_stochastic"] = path

        dist_g = distribution_table(
            enriched.loc[enriched["is_greedy"].astype(bool)],
            col,
            stochastic_only=False,
        )
        path = out_dir / f"dist_{col}_greedy.csv"
        dist_g.to_csv(path, index=False)
        paths[f"dist_{col}_greedy"] = path

    # Slim enriched snapshot for downstream notebooks
    slim_cols = [
        c
        for c in [
            "request_id",
            "order_id",
            "temperatura",
            "repeticao",
            "is_greedy",
            "graduacao_codigo",
            "parse_status",
            "order_match",
            "requested_key_order",
            "observed_key_order",
            *_DEMOGRAPHIC_KEYS,
        ]
        if c in enriched.columns
    ]
    slim = enriched[slim_cols]
    path = out_dir / "enriched_samples.parquet"
    slim.to_parquet(path, index=False)
    paths["enriched_samples"] = path

    return paths
