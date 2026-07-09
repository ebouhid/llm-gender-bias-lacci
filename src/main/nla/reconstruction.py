"""NLA critic (activation reconstructor) for round-trip scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")


class NLACritic:
    """Load an NLA critic and compute reconstruction MSE."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        meta = yaml.safe_load((checkpoint_dir / "nla_meta.yaml").read_text())
        assert meta["role"] in ("critic", "ar"), (
            f"sidecar role={meta['role']!r}, expected 'critic' or 'ar'."
        )
        ms = meta["extraction"]["mse_scale"]
        assert ms is not None, (
            "sidecar mse_scale is None; NLACritic.score() requires a numeric mse_scale."
        )
        self.mse_scale: float = float(ms)
        self.template: str = meta["prompt_templates"].get("ar") or meta["prompt_templates"][
            "critic"
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )
        probe = self.tokenizer("x", add_special_tokens=True)["input_ids"]
        bos = self.tokenizer.bos_token_id
        assert bos is None or probe[0] == bos

        backbone = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir), torch_dtype=dtype, trust_remote_code=True
        )
        backbone.lm_head = torch.nn.Identity()
        inner = backbone.model
        for attr in _FINAL_LN_ATTRS:
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        else:
            raise AssertionError(
                f"no final-LN attribute on {type(inner).__name__} — tried "
                f"{_FINAL_LN_ATTRS!r}."
            )

        d = backbone.config.hidden_size
        self.value_head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
        head_path = checkpoint_dir / "value_head.safetensors"
        assert head_path.exists(), f"no value_head.safetensors at {checkpoint_dir!r}"
        self.value_head.load_state_dict(load_file(str(head_path)))

        self.backbone = backbone.to(device).eval()
        self.value_head = self.value_head.to(device).eval()
        self.device = device
        print(
            f"[NLACritic] {backbone.config.num_hidden_layers} layers  "
            f"d_model={d}  mse_scale={self.mse_scale:.2f}"
        )

    @torch.inference_mode()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        prompt = self.template.format(explanation=explanation)
        ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        )["input_ids"].to(self.device)
        h = self.backbone.model(ids, use_cache=False).last_hidden_state[0, -1]
        return self.value_head(h).float().cpu()

    def score(
        self, explanation: str, original: np.ndarray | torch.Tensor
    ) -> tuple[float, float]:
        pred = self.reconstruct(explanation)
        gold = torch.as_tensor(np.asarray(original, dtype=np.float32))
        pred_n = pred / pred.norm().clamp_min(1e-12) * self.mse_scale
        gold_n = gold / gold.norm().clamp_min(1e-12) * self.mse_scale
        mse = ((pred_n - gold_n) ** 2).mean().item()
        cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
        return mse, cos

    def score_with_norms(
        self, explanation: str, original: np.ndarray | torch.Tensor
    ) -> tuple[float, float, float, float]:
        """Return (mse, cosine, original_norm, reconstructed_norm)."""
        pred = self.reconstruct(explanation)
        gold = torch.as_tensor(np.asarray(original, dtype=np.float32))
        original_norm = gold.norm().item()
        reconstructed_norm = pred.norm().item()
        mse, cos = self.score(explanation, original)
        return mse, cos, original_norm, reconstructed_norm
