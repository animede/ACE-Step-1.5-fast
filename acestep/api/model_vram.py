"""Rough VRAM requirement per DiT model.

Used to refuse a model that cannot fit this GPU before any weights are read
(``/v1/init``) and to tell clients which models are selectable
(``/v1/models`` ``vram_ok``).  The numbers are the measured peaks of one
song with the LM 0.6B + text encoder + VAE around the DiT, rounded down a
little so a 12 GB card (reported as 11.x GB) still passes for the 2B models.
"""

from __future__ import annotations

from typing import Optional

# bf16 peaks: XL (4B) 300 s song ~13.6 GB, 2B turbo/base 60-300 s ~8.5 GB in the
# 12gb profile and ~6.5 GB with the LM / VAE offloaded (the 8 GB recipe).
_XL_BF16_GB = 14.0
_XL_INT8_GB = 9.0
_2B_BF16_GB = 6.0
_2B_INT8_GB = 4.5
# 12 GB cards report 11.9x GB, 8 GB cards 7.9x GB: allow that rounding.
_SLACK_GB = 0.5


def _quantized(quantization: Optional[str]) -> bool:
    q = str(quantization or "none").lower()
    return "int8" in q or "fp8" in q or "int4" in q


def is_xl_model(model_name: Optional[str]) -> bool:
    return "xl" in (model_name or "").lower()


def min_vram_gb(model_name: Optional[str], quantization: Optional[str] = None) -> float:
    """Smallest GPU (total VRAM, GB) that runs *model_name* with *quantization*."""
    if is_xl_model(model_name):
        return _XL_INT8_GB if _quantized(quantization) else _XL_BF16_GB
    return _2B_INT8_GB if _quantized(quantization) else _2B_BF16_GB


def vram_fits(model_name: Optional[str], gpu_memory_gb: float, quantization: Optional[str] = None) -> bool:
    """True when the GPU is big enough, or when its size is unknown (0)."""
    if gpu_memory_gb <= 0:
        return True
    return gpu_memory_gb + _SLACK_GB >= min_vram_gb(model_name, quantization)


def vram_problem(model_name: Optional[str], gpu_memory_gb: float, quantization: Optional[str] = None) -> Optional[str]:
    """Human-readable reason the model cannot be loaded, or ``None`` when it fits."""
    if vram_fits(model_name, gpu_memory_gb, quantization):
        return None
    need = min_vram_gb(model_name, quantization)
    hint = (
        "pick a 2B model (acestep-v15-turbo / -base / -sft) or start the server with ACESTEP_GPU_PROFILE=12gb"
        if is_xl_model(model_name)
        else "set ACESTEP_QUANTIZATION=int8_weight_only and offload the LM / VAE (see README, 8GB)"
    )
    return (
        f"VRAM too small for {model_name}: needs about {need:g} GB, this GPU has {gpu_memory_gb:.1f} GB. "
        f"Next step: {hint}."
    )
