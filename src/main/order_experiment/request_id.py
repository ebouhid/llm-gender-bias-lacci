"""Stable request identity for resumable, idempotent execution."""

from __future__ import annotations

import hashlib
import json


def make_request_id(
    *,
    modelo: str,
    provider: str,
    order_id: str,
    graduacao_codigo: str,
    temperatura: float,
    repeticao: int,
    prompt_version: str,
    is_greedy: bool,
    system_text: str,
    user_text: str,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    """Deterministic MD5 over the identity-relevant request fields."""
    payload = {
        "modelo": modelo,
        "provider": provider,
        "order_id": order_id,
        "graduacao_codigo": graduacao_codigo,
        "temperatura": float(temperatura),
        "top_k": None if top_k is None else int(top_k),
        "top_p": None if top_p is None else float(top_p),
        "repeticao": int(repeticao),
        "prompt_version": prompt_version,
        "is_greedy": bool(is_greedy),
        "system_text": system_text,
        "user_text": user_text,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()
