"""Load order conditions and render the Chaves block for prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.main.order_experiment.schemas import EXPECTED_KEYS, ORDER_IDS

_KEY_PLACEHOLDERS: dict[str, str] = {
    "nome": '"..."',
    "idade": "0",
    "estado": '"..."',
    "renda_mensal": "0",
    "sexo_atribuido": '"..."',
    "cor_ou_raca": '"..."',
}


def load_orders(path: str | Path) -> dict[str, list[str]]:
    """Load O0–O5 key orders from YAML. Keys must be a permutation of EXPECTED_KEYS."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "orders" not in raw:
        raise ValueError(f"orders file must contain top-level 'orders': {path}")
    orders: dict[str, list[str]] = {}
    expected = set(EXPECTED_KEYS)
    for order_id in ORDER_IDS:
        if order_id not in raw["orders"]:
            raise KeyError(f"missing order_id {order_id} in {path}")
        keys = [str(k) for k in raw["orders"][order_id]]
        if set(keys) != expected:
            raise ValueError(
                f"{order_id} must be a permutation of {EXPECTED_KEYS}, got {keys}"
            )
        if len(keys) != len(EXPECTED_KEYS):
            raise ValueError(f"{order_id} has duplicate keys: {keys}")
        orders[order_id] = keys
    return orders


def render_chaves_block(key_order: list[str]) -> str:
    """Render the prompt 'Chaves:' bullet list in the requested order."""
    lines = ["Chaves:"]
    for key in key_order:
        placeholder = _KEY_PLACEHOLDERS[key]
        lines.append(f'      - "{key}": {placeholder},')
    return "\n".join(lines)


def ordered_keys_as_json_list(key_order: list[str]) -> str:
    """Stable JSON-array string for Parquet string columns."""
    import json

    return json.dumps(key_order, ensure_ascii=False)
