"""Minimal NLA gender-bias analysis helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

_RACE_CODES = ("branca", "preta", "parda", "amarela", "indigena")
_DROP_FOR_MERGED = ("activation_vector", "nla_raw_response")


def parse_condition_gender(marcador_codigo: str | None) -> str:
    """Map marcador_codigo to masculino / feminino / sem_genero_declarado."""
    code = (marcador_codigo or "").strip().lower()
    if code.startswith("masculino"):
        return "masculino"
    if code.startswith("feminino"):
        return "feminino"
    return "sem_genero_declarado"


def parse_condition_race(marcador_codigo: str | None) -> str:
    """Map marcador_codigo to IBGE-style race/color or sem_cor_declarada."""
    code = (marcador_codigo or "").strip().lower().replace("indígena", "indigena")
    for race in _RACE_CODES:
        if code == race or code.endswith(f"_{race}"):
            return race
    return "sem_cor_declarada"


def _format_prompt(system: dict, user: dict) -> str:
    """Build a short display prompt from nested generation JSONL fields."""
    marcador = system.get("marcador_descricao") or system.get("marcador") or ""
    disciplina = user.get("disciplina_descricao") or user.get("disciplina") or ""
    system_line = f"Eu estou no último ano do ensino médio{marcador}."
    user_line = str(disciplina).strip()
    if user_line:
        return f"{system_line}\n{user_line}"
    return system_line


def _parse_jsonl_generation(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            system = record.get("system_prompt", {}) or {}
            user = record.get("user_prompt", {}) or {}
            if not isinstance(system, dict):
                system = {}
            if not isinstance(user, dict):
                user = {}
            rows.append(
                {
                    "modelo": record.get("modelo"),
                    "temperatura": record.get("temperatura"),
                    "repeticao": record.get("repeticao"),
                    "marcador_codigo": system.get("marcador_codigo"),
                    "disciplina_codigo": user.get("disciplina_codigo"),
                    "prompt": _format_prompt(system, user),
                    "model_output": record.get("resposta_raw"),
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


def enrich_condition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add condition_gender / condition_race parsed from marcador_codigo."""
    out = df.copy()
    if "marcador_codigo" in out.columns:
        out["condition_gender"] = out["marcador_codigo"].map(parse_condition_gender)
        out["condition_race"] = out["marcador_codigo"].map(parse_condition_race)
    return out


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

    return enrich_condition_columns(joined)


def prepare_merged_results(joined: pd.DataFrame) -> pd.DataFrame:
    """Drop heavy/raw columns for the persisted visualization table."""
    out = enrich_condition_columns(joined)
    drop_cols = [c for c in _DROP_FOR_MERGED if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def write_merged_results(path: str | Path, joined: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = prepare_merged_results(joined)
    merged.to_parquet(path, index=False)
    return path


def read_merged_results(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


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


def tokenize_explanation(text: str) -> list[str]:
    """Simple word tokenizer for keyword summaries."""
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"[a-zA-ZÀ-ÿ0-9']+", text.lower())
