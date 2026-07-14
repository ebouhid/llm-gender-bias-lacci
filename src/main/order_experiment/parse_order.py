"""Extract observed JSON key order from raw model output without mutating it."""

from __future__ import annotations

import json
import re
from typing import Any

from src.main.order_experiment.schemas import (
    PARSE_STATUS_EMPTY,
    PARSE_STATUS_MALFORMED,
    PARSE_STATUS_OK,
)

_KEY_PATTERN = re.compile(r'"((?:\\.|[^"\\])*)"\s*:')


def extract_json_object_span(text: str) -> str | None:
    """Return the substring from the first '{' to the last '}' if both exist."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _keys_via_regex(blob: str) -> list[str]:
    return [m.group(1) for m in _KEY_PATTERN.finditer(blob)]


def parse_observed_key_order(resposta_raw: str | None) -> dict[str, Any]:
    """Parse observed key order and status from raw text.

    Never mutates or rewrites ``resposta_raw``. Malformed outputs are labeled,
    not retried by the caller.
    """
    if resposta_raw is None or str(resposta_raw).strip() == "":
        return {
            "observed_key_order": None,
            "parse_status": PARSE_STATUS_EMPTY,
            "parsed_obj": None,
        }

    blob = extract_json_object_span(str(resposta_raw))
    if blob is None:
        return {
            "observed_key_order": None,
            "parse_status": PARSE_STATUS_MALFORMED,
            "parsed_obj": None,
        }

    try:
        parsed_obj = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        observed = _keys_via_regex(blob)
        return {
            "observed_key_order": observed or None,
            "parse_status": PARSE_STATUS_MALFORMED,
            "parsed_obj": None,
        }

    if not isinstance(parsed_obj, dict):
        return {
            "observed_key_order": None,
            "parse_status": PARSE_STATUS_MALFORMED,
            "parsed_obj": None,
        }

    observed = [str(k) for k in parsed_obj.keys()]
    return {
        "observed_key_order": observed,
        "parse_status": PARSE_STATUS_OK,
        "parsed_obj": parsed_obj,
    }


def compare_orders(
    requested: list[str] | str | None,
    observed: list[str] | str | None,
) -> bool | None:
    """Return True/False if both orders are available, else None."""
    req = _coerce_key_list(requested)
    obs = _coerce_key_list(observed)
    if req is None or obs is None:
        return None
    return req == obs


def _coerce_key_list(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return [str(v) for v in loaded]
        except json.JSONDecodeError:
            return [part.strip() for part in text.split(",") if part.strip()]
    return None


def annotate_sample_parse(
    *,
    resposta_raw: str | None,
    requested_key_order: list[str] | str,
) -> dict[str, Any]:
    """Build parse fields for a sample row (does not include resposta_raw)."""
    parsed = parse_observed_key_order(resposta_raw)
    observed = parsed["observed_key_order"]
    order_match = compare_orders(requested_key_order, observed)
    observed_json = (
        json.dumps(observed, ensure_ascii=False) if observed is not None else None
    )
    return {
        "observed_key_order": observed_json,
        "order_match": order_match,
        "parse_status": parsed["parse_status"],
        "parsed_obj": parsed["parsed_obj"],
    }
