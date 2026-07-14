"""Provider wrapper: call local OpenAI-compatible API and classify infra errors."""

from __future__ import annotations

import asyncio
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError


class InfrastructureError(Exception):
    """Transient / transport failures that may be retried."""

    def __init__(self, message: str, *, error_class: str):
        super().__init__(message)
        self.error_class = error_class


_INFRA_TYPES = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    OSError,
)


def _classify_exception(exc: BaseException) -> tuple[bool, str]:
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status is not None and int(status) >= 500:
            return True, f"APIStatusError_{status}"
        return False, f"APIStatusError_{status}"
    for typ in _INFRA_TYPES:
        if isinstance(exc, typ):
            return True, type(exc).__name__
    # httpx errors if surfaced without openai wrap
    name = type(exc).__name__
    if name in {"ConnectError", "ReadTimeout", "WriteTimeout", "RemoteProtocolError"}:
        return True, name
    return False, name


async def call_model_raw(
    *,
    provider: str,
    modelo: str,
    temperatura: float,
    system_prompt: str,
    user_prompt: str,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    """Return the exact model string. Raises InfrastructureError on infra faults.

    Empty or malformed content is returned as-is (or empty string); callers must
    not treat that as retryable infrastructure failure.
    """
    # Lazy import: keeps planning/analysis usable without API client deps loaded.
    from src.main.utils import chamar_api_provider

    try:
        response = await chamar_api_provider(
            provider,
            modelo,
            temperatura,
            system_prompt,
            user_prompt,
            top_k=top_k,
            top_p=top_p,
        )
    except Exception as exc:  # noqa: BLE001 — classify then re-raise
        is_infra, error_class = _classify_exception(exc)
        if is_infra:
            raise InfrastructureError(str(exc), error_class=error_class) from exc
        raise

    if response is None:
        return ""
    return response if isinstance(response, str) else str(response)


def sample_row_from_request(
    request: dict[str, Any],
    *,
    resposta_raw: str | None,
    parse_status: str,
    observed_key_order: str | None,
    order_match: bool | None,
    error_class: str | None,
    attempts: int,
    completed_at: str,
) -> dict[str, Any]:
    row = {k: request[k] for k in request.keys()}
    row.update(
        {
            "resposta_raw": resposta_raw,
            "observed_key_order": observed_key_order,
            "order_match": order_match,
            "parse_status": parse_status,
            "error_class": error_class,
            "attempts": attempts,
            "completed_at": completed_at,
        }
    )
    return row
