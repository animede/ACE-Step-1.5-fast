"""Plain INT8 weight-only Linear for the DiT decoder.

Per-output-channel symmetric INT8 weights with a bf16 scale, dequantized to the
activation dtype inside ``forward``. Unlike torchao's ``int8_weight_only`` this
is an ordinary ``nn.Module`` (no tensor subclass), so ``torch.compile`` can fuse
the dequantization with the surrounding elementwise work instead of falling
into a slow lowering (torchao int8wo + compile stalled at 42 s/step on an RTX
A4000; this module + compile runs 18% faster than uncompiled torchao int8wo
with the same VRAM and output quality, see docs/ja/低VRAM_XL_プローブ_20261116.md).
"""

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8PlainLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` holding INT8 weights."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.detach()
        scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        quantized = torch.round(weight.float() / scale[:, None].float()).clamp(-127, 127).to(torch.int8)
        self.register_buffer("weight_int8", quantized)
        self.register_buffer("scale", scale.to(weight.dtype))
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)[:, None]
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8"


def quantize_linears(model: nn.Module, *, root: str = "decoder", skip: Iterable[str] = ("tokenizer", "detokenizer")) -> int:
    """Replace every ``nn.Linear`` under ``root`` (except ``skip`` subtrees) and return the count."""
    skip = tuple(skip)
    replaced = 0
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        if not parts or parts[0] != root or any(part in skip for part in parts):
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, child_name, Int8PlainLinear(child))
                replaced += 1
    return replaced
