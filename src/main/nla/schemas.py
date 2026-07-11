"""Column names and dataclasses for NLA pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Activation parquet columns
ACTIVATION_COLUMNS: Final[tuple[str, ...]] = (
    "activation_id",
    "run_id",
    "example_id",
    "condition_id",
    "modelo",
    "temperatura",
    "repeticao",
    "marcador_codigo",
    "disciplina_codigo",
    "prompt_hash",
    "system_text",
    "user_text",
    "response_text",
    "base_model",
    "nla_layer",
    "token_index",
    "token_id",
    "token_str",
    "token_role",
    "char_start",
    "char_end",
    "activation_vector",
)

# Verbalization parquet columns
VERBALIZATION_COLUMNS: Final[tuple[str, ...]] = (
    "activation_id",
    "nla_av_checkpoint",
    "nla_explanation",
    "nla_raw_response",
    "temperature",
    "max_new_tokens",
    "verbalized_at",
)

# Reconstruction parquet columns
RECONSTRUCTION_COLUMNS: Final[tuple[str, ...]] = (
    "activation_id",
    "nla_ar_checkpoint",
    "reconstruction_mse",
    "reconstruction_cosine",
    "original_norm",
    "reconstructed_norm",
)

TOKEN_ROLES: Final[tuple[str, ...]] = (
    "special_token",
    "system_prompt",
    "user_prompt",
    "demographic_marker",
    "task_instruction",
    "answer_space",
    "generated_output",
)

VERBALIZATION_TIERS: Final[tuple[str, ...]] = (
    "tier1",
    "tier2",
    "tier3",
    "tier4",
)


@dataclass(frozen=True)
class ExampleContext:
    """Metadata for one expanded prompt example."""

    run_id: str
    example_id: str
    condition_id: str
    modelo: str
    temperatura: float
    repeticao: int
    marcador_codigo: str
    disciplina_codigo: str
    prompt_hash: str
    system_text: str
    user_text: str
    marcador_descricao: str
    response_text: str | None = None
