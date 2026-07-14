"""Build the immutable request grid for the attribute-order experiment."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from src.main.order_experiment.orders import load_orders, ordered_keys_as_json_list
from src.main.order_experiment.prompts import build_user_prompt
from src.main.order_experiment.request_id import make_request_id
from src.main.order_experiment.schemas import (
    FULL_STOCHASTIC_REPS,
    ORDER_IDS,
    PILOT_STOCHASTIC_REPS,
    REQUEST_COLUMNS,
)


def _as_path(root: Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def load_fields_from_source(path: Path) -> list[dict[str, str]]:
    """Load graduacao list from undergraduate_fields_for_profile-style YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "graduacao" in raw[0]:
        items = raw[0]["graduacao"]
    elif isinstance(raw, dict) and "graduacao" in raw:
        items = raw["graduacao"]
    elif isinstance(raw, dict) and "fields" in raw:
        items = raw["fields"]
    else:
        raise ValueError(f"unrecognized fields format: {path}")
    return [
        {"codigo": str(item["codigo"]), "descricao": str(item["descricao"])}
        for item in items
        if item.get("descricao") is not None
    ]


def load_fields_from_profile_config(path: Path) -> list[dict[str, str]]:
    """Load the 87-field profile experiment grid from profile_config.yaml."""
    cfg = OmegaConf.load(path)
    graduacao = cfg.CHAVES_PROMPT[0].graduacao
    fields: list[dict[str, str]] = []
    for item in graduacao:
        descricao = item.get("descricao") if hasattr(item, "get") else item.descricao
        if descricao is None:
            continue
        codigo = item.get("codigo") if hasattr(item, "get") else item.codigo
        fields.append({"codigo": str(codigo), "descricao": str(descricao)})
    return fields


def load_pilot_fields(path: Path) -> list[dict[str, str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw["fields"] if isinstance(raw, dict) and "fields" in raw else raw
    return [
        {"codigo": str(item["codigo"]), "descricao": str(item["descricao"])}
        for item in items
    ]


def resolve_fields(cfg: DictConfig, repo_root: Path) -> list[dict[str, str]]:
    mode = str(cfg.EXPERIMENT_MODE).lower()
    if mode == "pilot":
        return load_pilot_fields(_as_path(repo_root, cfg.FIELDS_PILOT_FILE))
    if mode == "full":
        return load_fields_from_profile_config(_as_path(repo_root, cfg.FIELDS_FULL_CONFIG))
    raise ValueError(f"EXPERIMENT_MODE must be 'pilot' or 'full', got {mode!r}")


def resolve_stochastic_reps(cfg: DictConfig) -> int:
    explicit = cfg.get("REPETICOES_STOCHASTIC", None)
    if explicit is not None:
        return int(explicit)
    mode = str(cfg.EXPERIMENT_MODE).lower()
    if mode == "pilot":
        return PILOT_STOCHASTIC_REPS
    if mode == "full":
        return FULL_STOCHASTIC_REPS
    raise ValueError(f"unknown EXPERIMENT_MODE: {mode}")


def resolve_top_k(cfg: DictConfig) -> int | None:
    """Return int top_k or None when unset / null in config."""
    value = cfg.get("TOP_K", None)
    if value is None:
        return None
    # OmegaConf may surface null as None already; also accept empty string.
    if value == "" or str(value).lower() in {"null", "none"}:
        return None
    return int(value)


def resolve_top_p(cfg: DictConfig) -> float | None:
    """Return float top_p or None when unset / null in config."""
    value = cfg.get("TOP_P", None)
    if value is None:
        return None
    if value == "" or str(value).lower() in {"null", "none"}:
        return None
    return float(value)


def expected_cardinality(
    *,
    n_fields: int,
    n_orders: int,
    n_temps: int,
    n_reps: int,
    run_greedy: bool,
    n_greedy_reps: int = 1,
) -> dict[str, int]:
    stochastic = n_fields * n_orders * n_temps * n_reps
    greedy = n_fields * n_orders * n_greedy_reps if run_greedy else 0
    return {
        "stochastic": stochastic,
        "greedy": greedy,
        "total": stochastic + greedy,
    }


def build_request_plan(cfg: DictConfig, repo_root: Path | None = None) -> pd.DataFrame:
    """Cartesian product of fields × orders × temperatures × repetitions."""
    root = repo_root or Path.cwd()
    fields = resolve_fields(cfg, root)
    orders = load_orders(_as_path(root, cfg.ORDERS_FILE))
    modelo_name, provider = cfg.MODELO
    system_text = str(cfg.get("SYSTEM_PROMPT", "") or "")
    prompt_version = str(cfg.PROMPT_VERSION)
    run_id = str(cfg.RUN_ID)
    experiment_mode = str(cfg.EXPERIMENT_MODE).lower()

    stochastic_temps = [float(t) for t in cfg.TEMPERATURES_STOCHASTIC]
    stochastic_reps = resolve_stochastic_reps(cfg)
    top_k = resolve_top_k(cfg)
    top_p = resolve_top_p(cfg)
    run_greedy = bool(cfg.get("RUN_GREEDY", True))
    greedy_temp = float(cfg.get("TEMPERATURE_GREEDY", 0.0))
    greedy_reps = int(cfg.get("REPETICOES_GREEDY", 1))

    cells: list[tuple[float, int, bool]] = []
    for temp in stochastic_temps:
        for rep in range(1, stochastic_reps + 1):
            cells.append((temp, rep, False))
    if run_greedy:
        for rep in range(1, greedy_reps + 1):
            cells.append((greedy_temp, rep, True))

    rows: list[dict[str, Any]] = []
    for field, order_id, (temperatura, repeticao, is_greedy) in itertools.product(
        fields, ORDER_IDS, cells
    ):
        key_order = orders[order_id]
        user_text = build_user_prompt(
            graduacao_descricao=field["descricao"],
            key_order=key_order,
        )
        request_id = make_request_id(
            modelo=str(modelo_name),
            provider=str(provider),
            order_id=order_id,
            graduacao_codigo=field["codigo"],
            temperatura=temperatura,
            repeticao=repeticao,
            prompt_version=prompt_version,
            is_greedy=is_greedy,
            system_text=system_text,
            user_text=user_text,
            top_k=top_k,
            top_p=top_p,
        )
        rows.append(
            {
                "request_id": request_id,
                "run_id": run_id,
                "experiment_mode": experiment_mode,
                "modelo": str(modelo_name),
                "provider": str(provider),
                "order_id": order_id,
                "requested_key_order": ordered_keys_as_json_list(key_order),
                "graduacao_codigo": field["codigo"],
                "graduacao_descricao": field["descricao"],
                "temperatura": temperatura,
                "top_k": top_k,
                "top_p": top_p,
                "repeticao": repeticao,
                "is_greedy": is_greedy,
                "prompt_version": prompt_version,
                "system_text": system_text,
                "user_text": user_text,
            }
        )

    df = pd.DataFrame(rows, columns=list(REQUEST_COLUMNS))
    if df["request_id"].duplicated().any():
        n_dup = int(df["request_id"].duplicated().sum())
        raise RuntimeError(f"request_id collisions in plan: {n_dup}")

    expected = expected_cardinality(
        n_fields=len(fields),
        n_orders=len(ORDER_IDS),
        n_temps=len(stochastic_temps),
        n_reps=stochastic_reps,
        run_greedy=run_greedy,
        n_greedy_reps=greedy_reps,
    )
    if len(df) != expected["total"]:
        raise RuntimeError(
            f"plan cardinality mismatch: got {len(df)}, expected {expected}"
        )
    return df
