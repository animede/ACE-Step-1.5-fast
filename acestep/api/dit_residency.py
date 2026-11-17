"""Exclusive GPU residency for multiple DiT handlers.

With ``ACESTEP_DIT_RESIDENCY=exclusive`` the API server keeps only one DiT on
the GPU at a time. The other handlers' DiT weights (``handler.model`` plus the
``silence_latent`` tensor) are parked in pinned host memory, so switching the
active model costs a PCIe copy (about one second for a bf16 XL DiT) instead of
a reload from disk, and the active model has the whole GPU for activations.

The queue executor serialises generations, so ``activate`` is only ever called
from the single generation thread; no locking is needed.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
from loguru import logger

RESIDENCY_ENV = "ACESTEP_DIT_RESIDENCY"


def residency_mode_from_env(value: Optional[str]) -> str:
    """Return ``"exclusive"`` or ``"shared"`` (the default) for an env value."""
    normalized = (value or "").strip().lower()
    return "exclusive" if normalized in {"exclusive", "swap", "1", "true"} else "shared"


_MASTER_ATTR = "_dit_residency_masters"


def _named_tensors(model: torch.nn.Module):
    yield from model.named_parameters()
    yield from model.named_buffers()


def _park_to_masters(handler: Any, model: torch.nn.Module) -> int:
    """Point every parameter/buffer at a persistent pinned host copy and drop the GPU copy.

    Inference never changes the weights, so the host copy made on the first park
    stays valid: later parks are just a re-point (no device-to-host copy, no
    re-pinning), which is what makes the swap cheap.
    """
    masters = getattr(handler, _MASTER_ATTR, None)
    if masters is None:
        masters = {}
        setattr(handler, _MASTER_ATTR, masters)
    copied = 0
    for name, tensor in list(_named_tensors(model)):
        master = masters.get(name)
        if master is None or master.shape != tensor.shape or master.dtype != tensor.dtype:
            host = tensor.data.detach().to("cpu")
            try:
                host = host.pin_memory()
            except (RuntimeError, NotImplementedError):
                pass  # e.g. torchao tensor subclasses: pageable copy, just slower to upload
            masters[name] = master = host
            copied += 1
        tensor.data = master
    return copied


class DitResidency:
    """Track which handler's DiT currently lives on the GPU and swap on demand."""

    def __init__(self, active: Any = None):
        self.active = active

    def prime(self, handler: Any) -> None:
        """Make the pinned host copies of ``handler``'s DiT now (it stays on the GPU).

        Doing this at startup moves the one-time host-copy cost (about 8 s for a
        bf16 XL DiT) out of the first model switch.
        """
        model = getattr(handler, "model", None)
        if model is None:
            return
        started = time.time()
        masters = getattr(handler, _MASTER_ATTR, None) or {}
        setattr(handler, _MASTER_ATTR, masters)
        for name, tensor in list(_named_tensors(model)):
            if name in masters:
                continue
            host = tensor.data.detach().to("cpu")
            try:
                host = host.pin_memory()
            except (RuntimeError, NotImplementedError):
                pass
            masters[name] = host
        logger.info("[dit_residency] primed host copies for {} in {:.2f}s",
                    getattr(handler, "model_name", type(handler).__name__), time.time() - started)

    def park(self, handler: Any) -> None:
        """Move ``handler``'s DiT to pinned host memory and free its VRAM."""
        model = getattr(handler, "model", None)
        if model is None:
            return
        started = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        from acestep.core.generation.handler.dit_fusions import drop_stacked_weights
        drop_stacked_weights(model)  # fused q/k/v and gate/up buffers would otherwise stay on the GPU
        copied = _park_to_masters(handler, model)
        silence = getattr(handler, "silence_latent", None)
        if silence is not None:
            handler.silence_latent = silence.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.active is handler:
            self.active = None
        logger.info(
            "[dit_residency] parked {} on CPU ({} host copies made) in {:.2f}s",
            getattr(handler, "model_name", type(handler).__name__), copied, time.time() - started,
        )

    def activate(self, handler: Any) -> None:
        """Ensure ``handler``'s DiT is on its device, parking the current one first."""
        if handler is None or self.active is handler:
            return
        if self.active is not None:
            self.park(self.active)
        model = getattr(handler, "model", None)
        device = getattr(handler, "device", "cuda")
        if model is None or str(device) == "cpu":
            self.active = handler
            return
        started = time.time()
        for _name, tensor in list(_named_tensors(model)):
            if tensor.device.type == "cpu":
                tensor.data = tensor.data.to(device, non_blocking=True)
        silence = getattr(handler, "silence_latent", None)
        if silence is not None:
            handler.silence_latent = silence.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.active = handler
        logger.info(
            "[dit_residency] activated {} on {} in {:.2f}s",
            getattr(handler, "model_name", type(handler).__name__), device, time.time() - started,
        )
