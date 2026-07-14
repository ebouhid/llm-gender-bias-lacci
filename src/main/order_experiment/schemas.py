"""Column contracts and constants for the JSON attribute-order experiment."""

from __future__ import annotations

from typing import Final

EXPECTED_KEYS: Final[tuple[str, ...]] = (
    "nome",
    "idade",
    "estado",
    "renda_mensal",
    "sexo_atribuido",
    "cor_ou_raca",
)

ORDER_IDS: Final[tuple[str, ...]] = (
    "O0_original",
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
)

PARSE_STATUS_OK: Final[str] = "ok"
PARSE_STATUS_MALFORMED: Final[str] = "malformed"
PARSE_STATUS_EMPTY: Final[str] = "empty"
PARSE_STATUS_INFRA_ERROR: Final[str] = "infra_error"

REQUEST_COLUMNS: Final[tuple[str, ...]] = (
    "request_id",
    "run_id",
    "experiment_mode",
    "modelo",
    "provider",
    "order_id",
    "requested_key_order",
    "graduacao_codigo",
    "graduacao_descricao",
    "temperatura",
    "top_k",
    "top_p",
    "repeticao",
    "is_greedy",
    "prompt_version",
    "system_text",
    "user_text",
)

SAMPLE_COLUMNS: Final[tuple[str, ...]] = REQUEST_COLUMNS + (
    "resposta_raw",
    "observed_key_order",
    "order_match",
    "parse_status",
    "error_class",
    "attempts",
    "completed_at",
)

PILOT_STOCHASTIC_REPS: Final[int] = 32
FULL_STOCHASTIC_REPS: Final[int] = 64
