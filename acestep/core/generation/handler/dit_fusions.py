"""Eager, shape-agnostic fusions for the XL DiT (no torch.compile needed).

Selected with ``ACESTEP_DIT_FUSIONS`` (comma list or ``all``):

- ``qkv``   one GEMM for q/k/v in self-attention (weights are views into one buffer, no extra VRAM)
- ``mlp``   one GEMM for gate/up in the SwiGLU MLP (same view trick)
- ``attn``  self-attention in [B, T, H, D] layout end to end: fused rotary
            (flash_attn.layers.rotary) and a direct flash_attn_func call, no
            transposes; falls back to the original forward when a mask is given
- ``adaln`` adaLN modulation and gated residuals as single addcmul kernels

All variants compute the same math as the original modules; differences are
bf16 last-bit rounding (GEMM tiling / fused kernels), the same class as the
fused RMSNorm. Measured on an RTX A4000 (see docs/ja/低VRAM_XL_プローブ_20261116.md).
"""

from __future__ import annotations

import os
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DIT_FUSIONS_ENV = "ACESTEP_DIT_FUSIONS"
_ALL = ("qkv", "mlp", "attn", "adaln")


def fusions_from_env(value: Optional[str]) -> tuple:
    names = tuple(part.strip().lower() for part in (value or "").split(",") if part.strip())
    if any(name in {"all", "1", "true"} for name in names):
        return _ALL
    return tuple(name for name in _ALL if name in names)


def _stacked_weight(module: nn.Module, attr: str, linears) -> torch.Tensor:
    """Return one buffer holding the concatenated weights of ``linears``, built lazily on the current device.

    The per-linear weights are re-pointed at slices of the buffer so no VRAM is
    duplicated. Rebuilt whenever the first weight no longer aliases the buffer
    (first call on the device, or after a CPU park / GPU activate re-pointed the
    parameters), so it follows device moves without extra bookkeeping.
    """
    buffer = getattr(module, attr, None)
    first = linears[0].weight
    if buffer is None or buffer.device != first.device or buffer.data_ptr() != first.data_ptr():
        buffer = torch.cat([lin.weight.data for lin in linears], dim=0)
        offset = 0
        for lin in linears:
            rows = lin.weight.shape[0]
            lin.weight.data = buffer[offset:offset + rows]
            offset += rows
        setattr(module, attr, buffer)
    return buffer


# ---------------------------------------------------------------- self-attention

def _attention_forward(self, hidden_states, attention_mask=None, past_key_value=None, cache_position=None,
                       encoder_hidden_states=None, position_embeddings=None, output_attentions=False, **kwargs):
    use_fast = (
        not (self.is_cross_attention and encoder_hidden_states is not None)
        and attention_mask is None
        and not output_attentions
        and position_embeddings is not None
        and hidden_states.is_cuda
    )
    if not use_fast:
        return self._acestep_orig_forward(
            hidden_states, attention_mask, past_key_value=past_key_value, cache_position=cache_position,
            encoder_hidden_states=encoder_hidden_states, position_embeddings=position_embeddings,
            output_attentions=output_attentions, **kwargs,
        )
    from flash_attn import flash_attn_func
    from flash_attn.layers.rotary import apply_rotary_emb

    bsz, seq_len, _ = hidden_states.shape
    head_dim = self.head_dim
    n_q = self.q_proj.weight.shape[0] // head_dim
    n_kv = self.k_proj.weight.shape[0] // head_dim
    if getattr(self, "_acestep_fuse_qkv", False):
        qkv_weight = _stacked_weight(self, "_acestep_qkv_weight", [self.q_proj, self.k_proj, self.v_proj])
        qkv = F.linear(hidden_states, qkv_weight)
        q, k, v = torch.split(qkv, [n_q * head_dim, n_kv * head_dim, n_kv * head_dim], dim=-1)
    else:
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
    q = self.q_norm(q.view(bsz, seq_len, n_q, head_dim))
    k = self.k_norm(k.view(bsz, seq_len, n_kv, head_dim))
    v = v.view(bsz, seq_len, n_kv, head_dim)

    cos, sin = position_embeddings  # [1 or B, T, head_dim] with duplicated halves (GPT-NeoX style)
    half = head_dim // 2
    cos_h, sin_h = cos[0, :, :half].contiguous(), sin[0, :, :half].contiguous()
    q = apply_rotary_emb(q, cos_h, sin_h, interleaved=False)
    k = apply_rotary_emb(k, cos_h, sin_h, interleaved=False)

    window = (-1, -1)
    sliding = self.sliding_window
    if sliding is not None and seq_len > sliding:
        window = (sliding - 1, sliding - 1)  # same as transformers' flash integration
    out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=self.scaling, causal=False, window_size=window)
    out = out.reshape(bsz, seq_len, n_q * head_dim)
    return self.o_proj(out), None


def fuse_attention(model: nn.Module, fuse_qkv_weights: bool) -> int:
    count = 0
    for module in model.modules():
        if type(module).__name__ != "AceStepAttention" or getattr(module, "is_cross_attention", False):
            continue
        if getattr(module, "_acestep_orig_forward", None) is None:
            module._acestep_orig_forward = module.forward
            module.forward = types.MethodType(_attention_forward, module)
        module._acestep_fuse_qkv = fuse_qkv_weights
        count += 1
    return count


# ---------------------------------------------------------------- MLP

def _mlp_forward(self, x):
    gate_up = F.linear(x, _stacked_weight(self, "_acestep_gate_up_weight", [self.gate_proj, self.up_proj]))
    gate, up = gate_up.chunk(2, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


def fuse_mlp(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3MLP" or getattr(module, "_acestep_mlp_fused", False):
            continue
        module._acestep_mlp_fused = True
        module.forward = types.MethodType(_mlp_forward, module)
        count += 1
    return count


# ---------------------------------------------------------------- DiT layer (adaLN glue)

def _dit_layer_forward(self, hidden_states, position_embeddings, temb, attention_mask=None, position_ids=None,
                       past_key_value=None, output_attentions=False, use_cache=False, cache_position=None,
                       encoder_hidden_states=None, encoder_attention_mask=None, **kwargs):
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.scale_shift_table + temb).chunk(6, dim=1)
    norm_hidden_states = torch.addcmul(shift_msa, self.self_attn_norm(hidden_states), scale_msa + 1).type_as(hidden_states)
    attn_output, self_attn_weights = self.self_attn(
        hidden_states=norm_hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
        position_ids=position_ids, output_attentions=output_attentions, use_cache=False, past_key_value=None, **kwargs,
    )
    hidden_states = torch.addcmul(hidden_states, attn_output, gate_msa).type_as(hidden_states)
    cross_attn_weights = None
    if self.use_cross_attention:
        norm_hidden_states = self.cross_attn_norm(hidden_states).type_as(hidden_states)
        attn_output, cross_attn_weights = self.cross_attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache, **kwargs,
        )
        hidden_states = hidden_states + attn_output
    norm_hidden_states = torch.addcmul(c_shift_msa, self.mlp_norm(hidden_states), c_scale_msa + 1).type_as(hidden_states)
    ff_output = self.mlp(norm_hidden_states)
    hidden_states = torch.addcmul(hidden_states, ff_output, c_gate_msa).type_as(hidden_states)
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)
    return outputs


def fuse_adaln(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if type(module).__name__ == "AceStepDiTLayer" and not getattr(module, "_acestep_adaln_fused", False):
            module.forward = types.MethodType(_dit_layer_forward, module)
            module._acestep_adaln_fused = True
            count += 1
    return count


STACKED_ATTRS = ("_acestep_qkv_weight", "_acestep_gate_up_weight")


def drop_stacked_weights(model: nn.Module) -> int:
    """Forget the concatenated weight buffers (rebuilt lazily on the next forward).

    Must be called before a DiT is parked on the CPU: the buffers hold most of
    the DiT's weights on the GPU and would otherwise leak VRAM after the
    per-parameter re-pointing done by ``DitResidency.park``.
    """
    dropped = 0
    for module in model.modules():
        for attr in STACKED_ATTRS:
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
                dropped += 1
    return dropped


def apply_dit_fusions(model: nn.Module, names: tuple) -> dict:
    """Apply the selected fusions to the DiT ``model``; return {name: modules touched}."""
    result = {}
    if "attn" in names:  # "qkv" only has an effect together with "attn"
        result["attn"] = fuse_attention(model, fuse_qkv_weights="qkv" in names)
    if "mlp" in names:
        result["mlp"] = fuse_mlp(model)
    if "adaln" in names:
        result["adaln"] = fuse_adaln(model)
    return result


def apply_dit_fusions_from_env(model: nn.Module) -> dict:
    names = fusions_from_env(os.environ.get(DIT_FUSIONS_ENV))
    return apply_dit_fusions(model, names) if names else {}
