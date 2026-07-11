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

    def _last_token_hidden(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(
            last_hidden_state.shape[0], device=last_hidden_state.device
        )
        return last_hidden_state[batch_idx, seq_lens]

    def _score_tensors(
        self, pred: torch.Tensor, gold: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_n = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.mse_scale
        gold_n = gold / gold.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.mse_scale
        mse = ((pred_n - gold_n) ** 2).mean(dim=-1)
        cos = (pred_n * gold_n).sum(dim=-1) / (
            pred_n.norm(dim=-1) * gold_n.norm(dim=-1)
        ).clamp_min(1e-12)
        return mse, cos

    @torch.inference_mode()
    def reconstruct_batch(self, explanations: list[str]) -> torch.Tensor:
        if not explanations:
            return torch.empty(0)

        prompts = [self.template.format(explanation=e) for e in explanations]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(self.device)
        h = self.backbone.model(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
        ).last_hidden_state
        last_h = self._last_token_hidden(h, encoded["attention_mask"])
        return self.value_head(last_h).float().cpu()

    @torch.inference_mode()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        return self.reconstruct_batch([explanation])[0]

    def score(
        self, explanation: str, original: np.ndarray | torch.Tensor
    ) -> tuple[float, float]:
        pred = self.reconstruct(explanation)
        gold = torch.as_tensor(np.asarray(original, dtype=np.float32))
        mse, cos = self._score_tensors(pred.unsqueeze(0), gold.unsqueeze(0))
        return mse.item(), cos.item()

    def score_with_norms(
        self, explanation: str, original: np.ndarray | torch.Tensor
    ) -> tuple[float, float, float, float]:
        """Return (mse, cosine, original_norm, reconstructed_norm)."""
        pred = self.reconstruct(explanation)
        gold = torch.as_tensor(np.asarray(original, dtype=np.float32))
        mse, cos = self._score_tensors(pred.unsqueeze(0), gold.unsqueeze(0))
        return (
            mse.item(),
            cos.item(),
            gold.norm().item(),
            pred.norm().item(),
        )

    def score_batch_with_norms(
        self,
        explanations: list[str],
        originals: list[np.ndarray | torch.Tensor],
    ) -> list[tuple[float, float, float, float]]:
        preds = self.reconstruct_batch(explanations)
        gold = torch.stack(
            [torch.as_tensor(np.asarray(o, dtype=np.float32)) for o in originals]
        )
        mse, cos = self._score_tensors(preds, gold)
        original_norms = gold.norm(dim=-1)
        reconstructed_norms = preds.norm(dim=-1)
        return [
            (
                mse[i].item(),
                cos[i].item(),
                original_norms[i].item(),
                reconstructed_norms[i].item(),
            )
            for i in range(len(explanations))
        ]
