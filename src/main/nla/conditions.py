"""Resolve recommendation vs profile condition keys into the NLA schema."""

from __future__ import annotations

from typing import Any


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return {}


def _str_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def resolve_condition_keys(
    system_keys: Any = None,
    user_keys: Any = None,
) -> tuple[str, str, str]:
    """Map experiment-specific keys onto marcador/disciplina columns.

    Recommendation runs use ``marcador_*`` / ``disciplina_*``.
    Profile runs use empty system keys and ``graduacao_*`` on the user prompt.

    Returns:
        ``(marcador_codigo, disciplina_codigo, marcador_descricao)``
    """
    system = _as_dict(system_keys)
    user = _as_dict(user_keys)

    marcador_codigo = _str_or_empty(system.get("marcador_codigo"))
    disciplina_codigo = _str_or_empty(
        user.get("disciplina_codigo") or user.get("graduacao_codigo")
    )
    marcador_descricao = _str_or_empty(system.get("marcador_descricao"))
    if not marcador_descricao:
        marcador_descricao = _str_or_empty(
            user.get("graduacao_descricao") or user.get("graduacao")
        )
    return marcador_codigo, disciplina_codigo, marcador_descricao
