"""Exclusive GPU residency for multiple DiT handlers.

With ``ACESTEP_DIT_RESIDENCY=exclusive`` the API server keeps only one DiT on
the GPU at a time. The other handlers' DiT weights (``handler.model`` plus the
``silence_latent`` tensor) are parked on the host through a store (pinned RAM
or a memory-mapped bf16 file, see ``dit_residency_store``), so switching the
active model costs a host-to-GPU copy (one to three seconds for a bf16 XL DiT)
instead of a reload from disk, and the active model has the whole GPU.

Concurrency (ACESTEP_QUEUE_WORKERS / ACESTEP_API_WORKERS > 1): generation
threads enter through ``use(handler)``. Jobs for the handler that is already
on the GPU run in parallel; a job for another handler waits until the in-flight
jobs finish, switches, and then runs. While a switch is waiting, new jobs for
the current handler also wait, so a steady stream of one model cannot starve
the other.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

import torch
from loguru import logger

from acestep.api.dit_residency_store import PinnedStore, named_tensors

RESIDENCY_ENV = "ACESTEP_DIT_RESIDENCY"


def residency_mode_from_env(value: Optional[str]) -> str:
    """Return ``"exclusive"`` or ``"shared"`` (the default) for an env value."""
    normalized = (value or "").strip().lower()
    return "exclusive" if normalized in {"exclusive", "swap", "1", "true"} else "shared"


def _label(handler: Any) -> str:
    params = getattr(handler, "last_init_params", None) or {}
    return str(params.get("config_path") or type(handler).__name__)


def _rebind(model: Any, name: str, tensor: torch.Tensor, new: torch.Tensor) -> None:
    """Point the parameter or buffer ``name`` of ``model`` at ``new``.

    Parameters keep their object (``.data`` swap, so optimizer-free inference code holding
    the Parameter still sees it). Buffers are re-registered instead: ``buffer.data = x``
    makes the buffer inherit the version-counter state of ``x``, and a buffer swapped that
    way through the mmap store failed later inside ``torch.inference_mode`` with
    "Inference tensors do not track version counter" (ResidualFSQ ``scales`` on the first
    thinking-mode decode after a model switch, 2026-11-22).
    """
    module_name, _, attr = name.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    if attr in module._parameters:
        tensor.data = new
    else:
        module._buffers[attr] = new


class DitResidency:
    """Track which handler's DiT currently lives on the GPU and swap on demand."""

    def __init__(self, active: Any = None, store: Any = None):
        self.active = active
        self.store = store or PinnedStore()
        self._cond = threading.Condition()
        self._in_use = 0
        self._switch_waiting = 0

    @contextmanager
    def use(self, handler: Any):
        """Hold ``handler`` on the GPU for the duration of a generation (thread-safe).

        Jobs on the active handler run concurrently. A job for another handler
        registers as a pending switch, waits for the in-flight jobs to finish,
        switches, and runs; jobs for the active handler arriving while a switch
        is pending wait behind it (no starvation of the other model).
        """
        with self._cond:
            counted = False
            try:
                while True:
                    if self.active is not handler:
                        if not counted:
                            self._switch_waiting += 1
                            counted = True
                        if self._in_use == 0:
                            self.activate(handler)
                            break
                    elif self._switch_waiting == 0 or counted:
                        break
                    self._cond.wait()
            finally:
                if counted:
                    self._switch_waiting -= 1
                    self._cond.notify_all()
            self._in_use += 1
        try:
            yield
        finally:
            with self._cond:
                self._in_use -= 1
                self._cond.notify_all()

    @contextmanager
    def exclusive(self):
        """Block new jobs and wait for in-flight ones; for admin operations (model switch / reload)."""
        with self._cond:
            self._switch_waiting += 1
            try:
                while self._in_use > 0:
                    self._cond.wait()
                yield
            finally:
                self._switch_waiting -= 1
                self._cond.notify_all()

    def prime(self, handler: Any) -> None:
        """Prepare the host side of ``handler``'s DiT now (it stays on the GPU).

        Moves the one-time cost (host copies, or writing the mmap file) out of
        the first model switch.
        """
        model = getattr(handler, "model", None)
        if model is None:
            return
        started = time.time()
        made = self.store.prepare(handler, model)
        logger.info("[dit_residency] primed {} ({} store, {} tensors) in {:.2f}s",
                    _label(handler), self.store.name, made, time.time() - started)

    def park(self, handler: Any) -> None:
        """Point ``handler``'s DiT at its host copies and free its VRAM."""
        model = getattr(handler, "model", None)
        if model is None:
            return
        started = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        from acestep.core.generation.handler.dit_fusions import drop_stacked_weights
        drop_stacked_weights(model)  # fused q/k/v and gate/up buffers would otherwise stay on the GPU
        masters = self.store.masters(handler, model)
        for name, tensor in list(named_tensors(model)):
            master = masters.get(name)
            if master is None or master.shape != tensor.shape:
                raise RuntimeError(f"[dit_residency] no host copy for {name} in the {self.store.name} store")
            _rebind(model, name, tensor, master)
        silence = getattr(handler, "silence_latent", None)
        if silence is not None:
            handler.silence_latent = silence.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.active is handler:
            self.active = None
        logger.info("[dit_residency] parked {} on CPU ({} store) in {:.2f}s", _label(handler), self.store.name, time.time() - started)

    def activate(self, handler: Any) -> None:
        """Ensure ``handler``'s DiT is on its device, parking the current one first."""
        if handler is None or self.active is handler:
            return
        if self.active is not None:
            self.park(self.active)
        model = getattr(handler, "model", None)
        device = getattr(handler, "device", "cuda")
        if model is None or str(device) == "cpu":
            self.store.release(handler)
            self.active = handler
            return
        started = time.time()
        for name, tensor in list(named_tensors(model)):
            if tensor.device.type == "cpu":
                _rebind(model, name, tensor, tensor.data.to(device, non_blocking=tensor.data.is_pinned()))
        silence = getattr(handler, "silence_latent", None)
        if silence is not None:
            handler.silence_latent = silence.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.store.release(handler)
        self.active = handler
        logger.info("[dit_residency] activated {} on {} in {:.2f}s", _label(handler), device, time.time() - started)
