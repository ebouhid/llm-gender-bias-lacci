"""Minimal NLA gender-bias analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _parse_jsonl_generation(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            system = record.get("system_prompt", {})
            user = record.get("user_prompt", {})
            rows.append(
                {
                    "modelo": record.get("modelo"),
                    "temperatura": record.get("temperatura"),
                    "repeticao": record.get("repeticao"),
                    "marcador_codigo": system.get("marcador_codigo"),
                    "disciplina_codigo": user.get("disciplina_codigo"),
                    "resposta_raw": record.get("resposta_raw"),
                }
            )
    return pd.DataFrame(rows)


def _extract_areas(resposta_raw: str) -> str:
    try:
        data = json.loads(resposta_raw)
        areas = data.get("areas_recomendadas", [])
        return " | ".join(areas) if isinstance(areas, list) else ""
    except (json.JSONDecodeError, TypeError):
        return ""


def build_joined_table(
    activations: pd.DataFrame,
    verbalizations: pd.DataFrame,
    reconstructions: pd.DataFrame,
    generation_jsonl: str | Path | None = None,
) -> pd.DataFrame:
    joined = activations.merge(verbalizations, on="activation_id", how="inner")
    joined = joined.merge(reconstructions, on="activation_id", how="inner")

    if generation_jsonl and Path(generation_jsonl).exists():
        gen = _parse_jsonl_generation(generation_jsonl)
        gen["areas_recomendadas"] = gen["resposta_raw"].map(_extract_areas)
        joined = joined.merge(
            gen,
            on=["modelo", "temperatura", "repeticao", "marcador_codigo", "disciplina_codigo"],
            how="left",
        )
    return joined


def token_local_contrast(joined: pd.DataFrame) -> pd.DataFrame:
    marker_rows = joined[joined["token_role"] == "demographic_marker"].copy()
    if marker_rows.empty:
        return pd.DataFrame()

    summary = (
        marker_rows.groupby(
            ["disciplina_codigo", "marcador_codigo"], as_index=False
        )
        .agg(
            n_tokens=("activation_id", "count"),
            mean_reconstruction_mse=("reconstruction_mse", "mean"),
            mean_reconstruction_cosine=("reconstruction_cosine", "mean"),
            mean_explanation_length=("nla_explanation", lambda s: s.str.len().mean()),
            sample_explanation=("nla_explanation", "first"),
        )
        .sort_values(["disciplina_codigo", "marcador_codigo"])
    )
    return summary


def span_level_summary(joined: pd.DataFrame) -> pd.DataFrame:
    return (
        joined.groupby("token_role", as_index=False)
        .agg(
            n_tokens=("activation_id", "count"),
            mean_reconstruction_mse=("reconstruction_mse", "mean"),
            mean_reconstruction_cosine=("reconstruction_cosine", "mean"),
            mean_explanation_length=("nla_explanation", lambda s: s.str.len().mean()),
        )
        .sort_values("token_role")
    )


def outcome_linked_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if "areas_recomendadas" not in joined.columns:
        return pd.DataFrame()

    marker_rows = joined[joined["token_role"] == "demographic_marker"].copy()
    if marker_rows.empty:
        return pd.DataFrame()

    return (
        marker_rows.groupby(
            ["marcador_codigo", "disciplina_codigo", "areas_recomendadas"],
            as_index=False,
        )
        .agg(
            n_tokens=("activation_id", "count"),
            mean_reconstruction_mse=("reconstruction_mse", "mean"),
            sample_explanation=("nla_explanation", "first"),
        )
        .sort_values(["disciplina_codigo", "marcador_codigo"])
    )
