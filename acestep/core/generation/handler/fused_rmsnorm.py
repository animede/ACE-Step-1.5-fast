"""Fused RMSNorm for the DiT (eager, shape-agnostic).

The DiT uses transformers' ``Qwen3RMSNorm`` (3 per layer plus q/k norms in every
attention): upcast to fp32, pow, mean, rsqrt, mul, cast back, mul weight — about
1900 small kernels per diffusion step, 21% of the step on an RTX A4000.
``torch.nn.functional.rms_norm`` runs one fused CUDA kernel (fp32 accumulation).

Modes (``ACESTEP_FUSED_RMSNORM``):
- ``exact``/``1``: ``weight * rms_norm(x)`` — same rounding order as Qwen3RMSNorm
  (normalize in fp32, round to the input dtype, then multiply by the weight).
- ``fast``: ``rms_norm(x, weight)`` — weight applied inside the kernel, one rounding.
Both differ from eager only by bf16 last-bit rounding (fp32 reduction order).
"""

from __future__ import annotations

import os
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSED_RMSNORM_ENV = "ACESTEP_FUSED_RMSNORM"


def fused_rmsnorm_mode(value: Optional[str]) -> Optional[str]:
    normalized = (value or "").strip().lower()
    if normalized in {"1", "true", "exact", "on"}:
        return "exact"
    if normalized == "fast":
        return "fast"
    return None


def _is_rmsnorm(module: nn.Module) -> bool:
    return (
        type(module).__name__.endswith("RMSNorm")
        and isinstance(getattr(module, "weight", None), torch.Tensor)
        and isinstance(getattr(module, "variance_epsilon", None), float)
        and module.weight.dim() == 1
    )


def _forward_exact(self, hidden_states: torch.Tensor) -> torch.Tensor:
    normalized = F.rms_norm(hidden_states, (hidden_states.shape[-1],), None, self.variance_epsilon)
    return self.weight * normalized


def _forward_fast(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(hidden_states, (hidden_states.shape[-1],), self.weight, self.variance_epsilon)


def fuse_rmsnorm(model: nn.Module, mode: str = "exact") -> int:
    """Replace the forward of every RMSNorm module under ``model``; return the count."""
    forward = _forward_fast if mode == "fast" else _forward_exact
    replaced = 0
    for module in model.modules():
        if _is_rmsnorm(module) and getattr(module, "_acestep_fused_rmsnorm", None) != mode:
            module.forward = types.MethodType(forward, module)
            module._acestep_fused_rmsnorm = mode
            replaced += 1
    return replaced


def fuse_rmsnorm_from_env(model: nn.Module) -> tuple:
    """Apply the env-selected mode (if any); return ``(mode, replaced)``."""
    mode = fused_rmsnorm_mode(os.environ.get(FUSED_RMSNORM_ENV))
    if mode is None:
        return None, 0
    return mode, fuse_rmsnorm(model, mode)
