"""Stable ID generation for NLA pipeline rows."""

from __future__ import annotations

import hashlib
from datetime import datetime


def make_run_id(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _hash_parts(*parts: object) -> str:
    payload = "|".join(str(p) for p in parts)
    return hashlib.md5(payload.encode()).hexdigest()


def make_prompt_hash(system_text: str, user_text: str) -> str:
    return _hash_parts(system_text, user_text)


def make_example_id(
    modelo: str,
    temperatura: float,
    repeticao: int,
    marcador_codigo: str,
    disciplina_codigo: str,
) -> str:
    return _hash_parts(
        modelo, temperatura, repeticao, marcador_codigo, disciplina_codigo
    )


def make_condition_id(marcador_codigo: str, disciplina_codigo: str) -> str:
    return f"{marcador_codigo}__{disciplina_codigo}"


def make_activation_id(
    run_id: str,
    example_id: str,
    base_model: str,
    nla_layer: int,
    token_index: int,
) -> str:
    return _hash_parts(run_id, example_id, base_model, nla_layer, token_index)
