"""Heuristic token-role annotation for gender-bias prompts."""

from __future__ import annotations

import re

ANSWER_SPACE_START = "Áreas disponíveis:"
TASK_INSTRUCTION_MARKERS = (
    "Quais áreas de formação",
    "Retorne somente um objeto JSON",
)


def _special_token_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    for attr in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    ):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            ids.add(int(val))
    for token in getattr(tokenizer, "all_special_tokens", []) or []:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        ids.update(int(x) for x in encoded)
    return ids


def _find_span(text: str, substring: str) -> tuple[int, int] | None:
    if not substring:
        return None
    start = text.find(substring)
    if start < 0:
        return None
    return start, start + len(substring)


def _overlaps(token_start: int, token_end: int, span_start: int, span_end: int) -> bool:
    return token_start < span_end and token_end > span_start


def _token_char_spans(tokenizer, input_ids: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token_id in input_ids:
        piece = tokenizer.decode([token_id])
        start = cursor
        end = cursor + len(piece)
        spans.append((start, end))
        cursor = end
    return spans


def annotate_token_roles(
    *,
    tokenizer,
    input_ids: list[int],
    system_text: str,
    user_text: str,
    marcador_descricao: str,
    response_text: str | None = None,
    prompt_token_count: int | None = None,
) -> list[dict]:
    """Return per-token metadata: token_id, token_str, token_role, char offsets."""
    full_text = tokenizer.decode(input_ids)
    char_spans = _token_char_spans(tokenizer, input_ids)

    system_span = _find_span(full_text, system_text)
    user_span = _find_span(full_text, user_text)
    marker_span = _find_span(full_text, marcador_descricao.strip()) if marcador_descricao else None
    response_span = _find_span(full_text, response_text) if response_text else None

    answer_space_span = None
    answer_idx = user_text.find(ANSWER_SPACE_START)
    if answer_idx >= 0 and user_span is not None:
        answer_space_span = (
            user_span[0] + answer_idx,
            user_span[1],
        )

    task_instruction_end = len(user_text)
    for marker in TASK_INSTRUCTION_MARKERS:
        idx = user_text.find(marker)
        if idx >= 0:
            task_instruction_end = min(task_instruction_end, idx)
    if answer_idx >= 0:
        task_instruction_end = min(task_instruction_end, answer_idx)
    task_instruction_span = None
    if user_span is not None:
        task_instruction_span = (
            user_span[0],
            user_span[0] + task_instruction_end,
        )

    special_ids = _special_token_ids(tokenizer)
    rows: list[dict] = []

    for token_index, token_id in enumerate(input_ids):
        token_str = tokenizer.decode([token_id])
        char_start, char_end = char_spans[token_index]

        if int(token_id) in special_ids or (
            token_str.strip() == "" and re.match(r"^<[^>]+>$", token_str)
        ):
            role = "special_token"
        elif marker_span and _overlaps(char_start, char_end, *marker_span):
            role = "demographic_marker"
        elif answer_space_span and _overlaps(char_start, char_end, *answer_space_span):
            role = "answer_space"
        elif task_instruction_span and _overlaps(
            char_start, char_end, *task_instruction_span
        ):
            role = "task_instruction"
        elif system_span and _overlaps(char_start, char_end, *system_span):
            role = "system_prompt"
        elif user_span and _overlaps(char_start, char_end, *user_span):
            role = "user_prompt"
        elif response_span and _overlaps(char_start, char_end, *response_span):
            role = "generated_output"
        elif prompt_token_count is not None and token_index >= prompt_token_count:
            role = "generated_output"
        else:
            role = "user_prompt"

        rows.append(
            {
                "token_index": token_index,
                "token_id": int(token_id),
                "token_str": token_str,
                "token_role": role,
                "char_start": int(char_start),
                "char_end": int(char_end),
            }
        )

    return rows
