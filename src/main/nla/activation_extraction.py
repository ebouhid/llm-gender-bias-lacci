"""Extract per-token residual activations from a local causal LM."""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.main.nla.ids import (
    make_activation_id,
    make_condition_id,
    make_example_id,
    make_prompt_hash,
)
from src.main.nla.schemas import ExampleContext
from src.main.nla.token_annotation import annotate_token_roles
from src.main.template_expansion import expandir_templates_v2

logger = logging.getLogger(__name__)

GenerationKey = tuple[str, float, int, str, str]


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported dtype: {dtype_name!r}")
    return mapping[dtype_name]


def load_generation_responses(path: str | Path | None) -> dict[GenerationKey, str]:
    """Load resposta_raw keyed by modelo/temperatura/repeticao/marcador/disciplina."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        logger.warning("GENERATION_JSONL not found: %s", path)
        return {}

    mapping: dict[GenerationKey, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            system = record.get("system_prompt", {}) or {}
            user = record.get("user_prompt", {}) or {}
            if not isinstance(system, dict):
                system = {}
            if not isinstance(user, dict):
                user = {}
            resposta = record.get("resposta_raw")
            if resposta is None:
                continue
            key: GenerationKey = (
                str(record.get("modelo") or ""),
                float(record.get("temperatura")),
                int(record.get("repeticao")),
                str(system.get("marcador_codigo") or ""),
                str(user.get("disciplina_codigo") or ""),
            )
            mapping[key] = str(resposta)
    logger.info("loaded %d generation responses from %s", len(mapping), path)
    return mapping


def build_example_contexts(
    cfg: Any,
    run_id: str,
    generation_responses: dict[GenerationKey, str] | None = None,
) -> list[ExampleContext]:
    system_expandidos = expandir_templates_v2(
        cfg.SYSTEM_PROMPT, cfg.CHAVES_SYSTEM_PROMPT
    )
    prompt_expandidos = expandir_templates_v2(cfg.PROMPTS, cfg.CHAVES_PROMPT)
    num_repeticoes = cfg.get("REPETICOES_POR_TEMP", 1)
    responses = generation_responses or {}

    contexts: list[ExampleContext] = []
    missing_responses = 0
    for system, prompt, modelo, temperatura, repeticao in itertools.product(
        system_expandidos,
        prompt_expandidos,
        cfg.MODELOS_A_AVALIAR,
        cfg.TEMPERATURES,
        range(1, num_repeticoes + 1),
    ):
        model_name, _provider = modelo
        system_text = system["texto"]
        user_text = prompt["texto"]
        system_keys = system["chaves_usadas"]
        user_keys = prompt["chaves_usadas"]

        marcador_codigo = system_keys.get("marcador_codigo", "")
        disciplina_codigo = user_keys.get("disciplina_codigo", "")
        marcador_descricao = system_keys.get("marcador_descricao", "")

        key: GenerationKey = (
            str(model_name),
            float(temperatura),
            int(repeticao),
            str(marcador_codigo),
            str(disciplina_codigo),
        )
        response_text = responses.get(key)
        if responses and response_text is None:
            missing_responses += 1

        example_id = make_example_id(
            model_name,
            float(temperatura),
            int(repeticao),
            marcador_codigo,
            disciplina_codigo,
        )
        contexts.append(
            ExampleContext(
                run_id=run_id,
                example_id=example_id,
                condition_id=make_condition_id(marcador_codigo, disciplina_codigo),
                modelo=model_name,
                temperatura=float(temperatura),
                repeticao=int(repeticao),
                marcador_codigo=marcador_codigo,
                disciplina_codigo=disciplina_codigo,
                prompt_hash=make_prompt_hash(system_text, user_text),
                system_text=system_text,
                user_text=user_text,
                marcador_descricao=marcador_descricao,
                response_text=response_text,
            )
        )
    if missing_responses:
        logger.warning(
            "%d example(s) missing a matching GENERATION_JSONL response",
            missing_responses,
        )
    return contexts


def _as_id_list(result) -> list[int]:
    if isinstance(result, list):
        return result
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def tokenize_prompt(
    tokenizer,
    system_text: str,
    user_text: str,
    response_text: str | None = None,
) -> list[int]:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    if response_text is not None:
        messages.append({"role": "assistant", "content": response_text})
        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    else:
        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    return _as_id_list(result)


def extract_activation_rows(
    *,
    model,
    tokenizer,
    context: ExampleContext,
    base_model: str,
    nla_layer: int,
) -> list[dict]:
    prompt_ids = tokenize_prompt(tokenizer, context.system_text, context.user_text)
    if context.response_text is not None:
        input_ids = tokenize_prompt(
            tokenizer,
            context.system_text,
            context.user_text,
            context.response_text,
        )
        prompt_token_count = len(prompt_ids)
    else:
        input_ids = prompt_ids
        prompt_token_count = None

    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)

    with torch.inference_mode():
        outputs = model(ids_tensor, output_hidden_states=True)
    hidden = outputs.hidden_states[nla_layer][0].float().cpu().numpy()

    token_meta = annotate_token_roles(
        tokenizer=tokenizer,
        input_ids=input_ids,
        system_text=context.system_text,
        user_text=context.user_text,
        marcador_descricao=context.marcador_descricao,
        response_text=context.response_text,
        prompt_token_count=prompt_token_count,
    )

    response_text = context.response_text if context.response_text is not None else ""
    rows: list[dict] = []
    for meta in token_meta:
        token_index = meta["token_index"]
        vector = hidden[token_index]
        rows.append(
            {
                "activation_id": make_activation_id(
                    context.run_id,
                    context.example_id,
                    base_model,
                    nla_layer,
                    token_index,
                ),
                "run_id": context.run_id,
                "example_id": context.example_id,
                "condition_id": context.condition_id,
                "modelo": context.modelo,
                "temperatura": context.temperatura,
                "repeticao": context.repeticao,
                "marcador_codigo": context.marcador_codigo,
                "disciplina_codigo": context.disciplina_codigo,
                "prompt_hash": context.prompt_hash,
                "system_text": context.system_text,
                "user_text": context.user_text,
                "response_text": response_text,
                "base_model": base_model,
                "nla_layer": nla_layer,
                "token_index": token_index,
                "token_id": meta["token_id"],
                "token_str": meta["token_str"],
                "token_role": meta["token_role"],
                "char_start": meta["char_start"],
                "char_end": meta["char_end"],
                "activation_vector": vector,
            }
        )
    return rows


def load_extraction_model(base_model: str, device: str, dtype_name: str):
    dtype = _resolve_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def extract_all_activations(
    cfg: Any,
    run_id: str,
    base_model: str,
    nla_layer: int,
    device: str,
    dtype_name: str,
    generation_jsonl: str | Path | None = None,
) -> list[dict]:
    path = generation_jsonl if generation_jsonl is not None else cfg.get("GENERATION_JSONL")
    responses = load_generation_responses(path)
    contexts = build_example_contexts(cfg, run_id, responses)
    model, tokenizer = load_extraction_model(base_model, device, dtype_name)

    all_rows: list[dict] = []
    for context in contexts:
        if context.modelo != base_model:
            continue
        rows = extract_activation_rows(
            model=model,
            tokenizer=tokenizer,
            context=context,
            base_model=base_model,
            nla_layer=nla_layer,
        )
        all_rows.extend(rows)
    return all_rows
