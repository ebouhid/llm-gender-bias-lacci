"""NLA actor (activation verbalizer) client via SGLang input_embeds."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson
import torch
import yaml
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
INJECT_PLACEHOLDER = "<INJECT>"
_EMBED_KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")


def _ids_from_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(result, list):
        return result
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


@dataclass(frozen=True)
class NLAConfig:
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    actor_prompt_template: str
    injection_scale: float


def load_nla_config(
    checkpoint_dir: str | Path,
    tokenizer: Any,
    injection_scale_override: float | None = None,
) -> NLAConfig:
    meta_path = Path(checkpoint_dir) / "nla_meta.yaml"
    assert meta_path.exists(), (
        f"no nla_meta.yaml at {checkpoint_dir!r}. Not an NLA checkpoint — "
        f"the sidecar ships alongside config.json/safetensors."
    )
    meta = yaml.safe_load(meta_path.read_text())

    kind = meta["kind"]
    assert kind in ("nla_model", "nla_dataset"), f"unknown sidecar kind: {kind!r}"
    d_model = meta["d_model"] if kind == "nla_model" else meta["extraction"]["d_model"]

    inj_scale = meta.get("extraction", {}).get("injection_scale")
    if inj_scale is None:
        inj_scale = injection_scale_override
    assert inj_scale is not None, (
        f"nla_meta.yaml at {checkpoint_dir!r} has no extraction.injection_scale "
        f"(kind={kind!r}, role={meta.get('role')!r})."
    )

    t = meta["tokens"]
    cfg = NLAConfig(
        d_model=d_model,
        injection_char=t["injection_char"],
        injection_token_id=t["injection_token_id"],
        injection_left_neighbor_id=t["injection_left_neighbor_id"],
        injection_right_neighbor_id=t["injection_right_neighbor_id"],
        actor_prompt_template=meta["prompt_templates"].get("av")
        or meta["prompt_templates"]["actor"],
        injection_scale=float(inj_scale),
    )

    live_inj = tokenizer.encode(cfg.injection_char, add_special_tokens=False)
    assert live_inj == [cfg.injection_token_id], (
        f"tokenizer drift: {cfg.injection_char!r} → {live_inj}, sidecar says "
        f"[{cfg.injection_token_id}]."
    )
    assert live_inj[0] != tokenizer.unk_token_id, (
        f"{cfg.injection_char!r} maps to UNK"
    )

    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    ids = _ids_from_chat_template(
        tokenizer, [{"role": "user", "content": content}]
    )
    matches = [i for i, tok in enumerate(ids) if tok == cfg.injection_token_id]
    assert len(matches) == 1, (
        f"injection token appears {len(matches)}× in canonical prompt "
        f"(expected 1). Template: {content!r}"
    )
    p = matches[0]
    assert 0 < p < len(ids) - 1
    assert ids[p - 1] == cfg.injection_left_neighbor_id
    assert ids[p + 1] == cfg.injection_right_neighbor_id

    return cfg


def load_embedding_only(
    checkpoint_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Embedding:
    root = Path(checkpoint_dir)

    def _find_key(keys: list[str], where: str) -> str:
        m = [k for k in keys if k.endswith(_EMBED_KEY_SUFFIXES)]
        assert len(m) == 1, (
            f"expected exactly one input-embedding key in {where}, got {m!r}"
        )
        return m[0]

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        key = _find_key(list(weight_map), str(index_path))
        shard = root / weight_map[key]
    else:
        shard = root / "model.safetensors"
        assert shard.exists(), f"no model.safetensors or .index.json at {root!r}"
        with safe_open(str(shard), framework="pt") as f:
            key = _find_key(list(f.keys()), str(shard))

    with safe_open(str(shard), framework="pt") as f:
        weight = f.get_tensor(key).to(dtype)

    vocab, d = weight.shape
    embed = torch.nn.Embedding(vocab, d, _weight=weight)
    embed.requires_grad_(False)
    embed.eval()
    return embed


_SCALED_EMBED_MODEL_TYPES = frozenset({
    "gemma", "gemma2", "gemma3", "gemma3_text", "t5",
})


def resolve_embed_scale(checkpoint_dir: str | Path) -> float:
    config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)
    model_type = getattr(text_cfg, "model_type", "") or ""
    if model_type in _SCALED_EMBED_MODEL_TYPES:
        return math.sqrt(text_cfg.hidden_size)
    return 1.0


def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
) -> torch.Tensor:
    seq_len = input_ids.shape[-1]
    assert input_ids.shape == embeddings.shape[:-1]
    assert vectors.ndim == 2 and vectors.shape[1] == embeddings.shape[-1]
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    vec_idx = 0
    for b, p in (input_ids == inj_id).nonzero().tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        out[b, p] = vectors[vec_idx]
        vec_idx += 1
    assert vec_idx == vectors.shape[0], (
        f"found {vec_idx} injection sites with correct neighbors, expected "
        f"{vectors.shape[0]}."
    )
    return out


class NLAClient:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        sglang_url: str = "http://localhost:30000",
        injection_scale_override: float | None = None,
        device: str = "cpu",
    ):
        checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )
        self.cfg = load_nla_config(
            checkpoint_dir,
            self.tokenizer,
            injection_scale_override=injection_scale_override,
        )

        self.embed = load_embedding_only(checkpoint_dir, dtype=torch.bfloat16).to(device)
        self.embed_scale = resolve_embed_scale(checkpoint_dir)

        assert self.embed.weight.shape[1] == self.cfg.d_model, (
            f"embedding d={self.embed.weight.shape[1]} != sidecar "
            f"d_model={self.cfg.d_model}."
        )

        self.sglang_url = sglang_url.rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(120.0))
        self._check_sglang_dtype(checkpoint_dir)

        print(
            f"[NLAClient] {checkpoint_dir.name}: d_model={self.cfg.d_model} "
            f"inj_scale={self.cfg.injection_scale} embed_scale={self.embed_scale:.2f} "
            f"inj_char={self.cfg.injection_char!r}(id={self.cfg.injection_token_id})"
        )

    def _check_sglang_dtype(self, checkpoint_dir: Path) -> None:
        """Fail fast if SGLang weights are float16 while input_embeds are forced to bf16."""
        try:
            resp = self._http.get(f"{self.sglang_url}/get_server_info")
            if resp.status_code != 200:
                return
            server_dtype = resp.json().get("dtype")
        except Exception:
            return

        if not isinstance(server_dtype, str):
            return
        if server_dtype.lower() == "bfloat16":
            return
        if server_dtype.lower() not in {"auto", "half", "float16"}:
            return

        hf_cfg = AutoConfig.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )
        torch_dtype = getattr(hf_cfg, "torch_dtype", None)
        if server_dtype.lower() == "auto" and torch_dtype not in (
            None,
            "float16",
            "half",
            torch.float16,
        ):
            return

        raise RuntimeError(
            "SGLang server dtype is incompatible with NLA input_embeds: "
            f"server dtype={server_dtype!r}, checkpoint torch_dtype={torch_dtype!r}. "
            "SGLang hardcodes input_embeds to bfloat16, so restart the actor "
            "server with --dtype bfloat16, e.g. `pixi run nla-sglang`."
        )

    def _prompt_content(self, prompt: str | None) -> str:
        if prompt is None:
            return self.cfg.actor_prompt_template.format(
                injection_char=self.cfg.injection_char
            )
        assert INJECT_PLACEHOLDER in prompt
        return prompt.replace(INJECT_PLACEHOLDER, self.cfg.injection_char)

    def _build_embeds_batch(
        self,
        activations: list[torch.Tensor],
        prompt: str | None = None,
    ) -> list[np.ndarray]:
        if not activations:
            return []

        content = self._prompt_content(prompt)
        input_ids = _ids_from_chat_template(
            self.tokenizer, [{"role": "user", "content": content}]
        )
        batch_size = len(activations)
        ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).expand(
            batch_size, -1
        )

        vectors = torch.stack(
            [
                torch.as_tensor(np.asarray(a, dtype=np.float32)).view(-1)
                for a in activations
            ]
        )
        assert vectors.shape == (batch_size, self.cfg.d_model), (
            f"expected vectors ({batch_size}, {self.cfg.d_model}), "
            f"got {tuple(vectors.shape)}"
        )
        assert torch.isfinite(vectors).all(), "activation has NaN/Inf"
        v_scaled = normalize_activation(vectors.float(), self.cfg.injection_scale)

        with torch.no_grad():
            embeds = (
                self.embed(ids_t.to(self.embed.weight.device)) * self.embed_scale
            ).float().cpu()

        injected = inject_at_marked_positions(
            ids_t,
            embeds,
            v_scaled,
            self.cfg.injection_token_id,
            self.cfg.injection_left_neighbor_id,
            self.cfg.injection_right_neighbor_id,
        )
        return [injected[i].contiguous().numpy() for i in range(batch_size)]

    def _build_embeds(
        self, v_raw: torch.Tensor, prompt: str | None
    ) -> tuple[np.ndarray, int]:
        content = self._prompt_content(prompt)
        input_ids = _ids_from_chat_template(
            self.tokenizer, [{"role": "user", "content": content}]
        )
        return self._build_embeds_batch([v_raw], prompt)[0], len(input_ids)

    def _sglang_generate(
        self,
        embeds: np.ndarray | list[np.ndarray],
        **sampling: object,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        is_batch = isinstance(embeds, list)
        sp = {"temperature": 1.0, "max_new_tokens": 200, "skip_special_tokens": False}
        sp.update(sampling)
        body = orjson.dumps(
            {"input_embeds": embeds, "sampling_params": sp},
            option=orjson.OPT_SERIALIZE_NUMPY,
        )
        resp = self._http.post(
            f"{self.sglang_url}/generate",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            detail = resp.text[:500]
            raise httpx.HTTPStatusError(
                f"SGLang /generate failed ({resp.status_code}): {detail}",
                request=resp.request,
                response=resp,
            )
        out = resp.json()
        if is_batch:
            return out if isinstance(out, list) else [out]
        return out[0] if isinstance(out, list) else out

    def _extract_explanation(self, text: str, extract_explanation: bool) -> str:
        if not extract_explanation:
            return text
        m = EXPLANATION_RE.search(text)
        if m is None:
            print(
                f"[NLAClient] WARNING: no <explanation> tags. "
                f"Raw[:200]={text[:200]!r}"
            )
            return text
        return m.group(1).strip()

    def generate(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        return_raw: bool = False,
        **sampling: object,
    ) -> str:
        v = torch.as_tensor(np.asarray(activation, dtype=np.float32))
        assert v.numel() == self.cfg.d_model, (
            f"activation length {v.numel()} != d_model {self.cfg.d_model}"
        )

        embeds_np, _ = self._build_embeds(v, prompt)
        out = self._sglang_generate(embeds_np, **sampling)
        text = out["text"]

        if return_raw:
            return text
        return self._extract_explanation(text, extract_explanation)

    def generate_with_raw(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        **sampling: object,
    ) -> tuple[str, str]:
        """Return (explanation, raw_response)."""
        v = torch.as_tensor(np.asarray(activation, dtype=np.float32))
        embeds_np, _ = self._build_embeds(v, prompt)
        out = self._sglang_generate(embeds_np, **sampling)
        raw = out["text"]
        return self._extract_explanation(raw, extract_explanation), raw

    def generate_batch(
        self,
        activations: Iterable[Iterable[float] | np.ndarray | torch.Tensor],
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        **sampling: object,
    ) -> list[str]:
        tensors = [
            torch.as_tensor(np.asarray(a, dtype=np.float32))
            for a in activations
        ]
        if not tensors:
            return []
        for v in tensors:
            assert v.numel() == self.cfg.d_model, (
                f"activation length {v.numel()} != d_model {self.cfg.d_model}"
            )

        embeds_list = self._build_embeds_batch(tensors, prompt)
        outs = self._sglang_generate(embeds_list, **sampling)
        assert len(outs) == len(tensors), (
            f"SGLang returned {len(outs)} outputs for batch of {len(tensors)}"
        )
        return [
            self._extract_explanation(out["text"], extract_explanation)
            for out in outs
        ]

    def generate_batch_with_raw(
        self,
        activations: Iterable[Iterable[float] | np.ndarray | torch.Tensor],
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        **sampling: object,
    ) -> list[tuple[str, str]]:
        tensors = [
            torch.as_tensor(np.asarray(a, dtype=np.float32))
            for a in activations
        ]
        if not tensors:
            return []
        for v in tensors:
            assert v.numel() == self.cfg.d_model, (
                f"activation length {v.numel()} != d_model {self.cfg.d_model}"
            )

        embeds_list = self._build_embeds_batch(tensors, prompt)
        outs = self._sglang_generate(embeds_list, **sampling)
        assert len(outs) == len(tensors), (
            f"SGLang returned {len(outs)} outputs for batch of {len(tensors)}"
        )
        return [
            (self._extract_explanation(out["text"], extract_explanation), out["text"])
            for out in outs
        ]
