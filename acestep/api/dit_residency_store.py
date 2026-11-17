"""Host-side stores for parked DiT weights (used by ``DitResidency``).

``PinnedStore`` keeps a pinned-RAM copy of every weight (fastest swap, but the
whole model stays in locked host memory). ``MmapStore`` writes each DiT once as
a bf16 safetensors file and parks by pointing the parameters at the memory-
mapped file: the weights then live in the OS page cache, which is evictable, so
the server's resident memory stays at its baseline (ACESTEP_DIT_RESIDENCY_STORE).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

import torch
from loguru import logger

STORE_ENV = "ACESTEP_DIT_RESIDENCY_STORE"
STORE_DIR_ENV = "ACESTEP_DIT_RESIDENCY_DIR"
_MASTERS_ATTR = "_dit_residency_masters"


def named_tensors(model: torch.nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
    """Parameters followed by buffers, with their qualified names."""
    yield from model.named_parameters()
    yield from model.named_buffers()


def _host_copy(tensor: torch.Tensor, pin: bool) -> torch.Tensor:
    host = tensor.data.detach().to("cpu")
    if pin:
        try:
            host = host.pin_memory()
        except (RuntimeError, NotImplementedError):
            pass  # e.g. torchao tensor subclasses: pageable copy, slower upload
    return host


class PinnedStore:
    """Persistent pinned host copies, made once per tensor and reused on every park."""

    name = "pinned"

    def prepare(self, handler: Any, model: torch.nn.Module) -> int:
        """Build the host copies now (model may stay on the GPU); return copies made."""
        masters = getattr(handler, _MASTERS_ATTR, None) or {}
        setattr(handler, _MASTERS_ATTR, masters)
        made = 0
        for name, tensor in list(named_tensors(model)):
            master = masters.get(name)
            if master is None or master.shape != tensor.shape or master.dtype != tensor.dtype:
                masters[name] = _host_copy(tensor, pin=True)
                made += 1
        return made

    def masters(self, handler: Any, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Host tensors to park onto (building any that are missing)."""
        self.prepare(handler, model)
        return getattr(handler, _MASTERS_ATTR)

    def release(self, handler: Any) -> None:
        """Pinned copies are kept for the next park."""


class MmapStore:
    """bf16 safetensors file per DiT, memory-mapped on park (page-cache backed)."""

    name = "mmap"

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def _file(self, handler: Any) -> str:
        params = getattr(handler, "last_init_params", None) or {}
        config = str(params.get("config_path") or "dit")
        return os.path.join(self.cache_dir, f"{os.path.basename(config.rstrip('/'))}.residency.safetensors")

    def prepare(self, handler: Any, model: torch.nn.Module) -> int:
        """Write the model's weights to the cache file unless a matching one exists; return tensors written."""
        from safetensors import safe_open
        from safetensors.torch import save_file

        path = self._file(handler)
        expected = {name: (tuple(t.shape), str(t.dtype)) for name, t in named_tensors(model)}
        if os.path.exists(path):
            try:
                with safe_open(path, framework="pt", device="cpu") as f:
                    found = {k: (tuple(f.get_slice(k).get_shape()), str(f.get_slice(k).get_dtype())) for k in f.keys()}
                if {k: v[0] for k, v in found.items()} == {k: v[0] for k, v in expected.items()}:
                    return 0
            except Exception as exc:  # unreadable or stale file: rewrite it
                logger.warning("[dit_residency] rewriting {}: {}", path, exc)
        os.makedirs(self.cache_dir, exist_ok=True)
        tensors = {name: _host_copy(t, pin=False).contiguous() for name, t in named_tensors(model)}
        save_file(tensors, path, metadata={"source": str((getattr(handler, "last_init_params", None) or {}).get("config_path"))})
        count = len(tensors)
        del tensors
        logger.info("[dit_residency] wrote {} ({} tensors)", path, count)
        return count

    def masters(self, handler: Any, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Memory-mapped host tensors (zero-copy; the file is written first if needed)."""
        from safetensors import safe_open

        self.prepare(handler, model)
        handle = safe_open(self._file(handler), framework="pt", device="cpu")
        masters = {name: handle.get_tensor(name) for name in handle.keys()}
        setattr(handler, _MASTERS_ATTR, (handle, masters))
        return masters

    def release(self, handler: Any) -> None:
        """Drop the mapping once the weights are back on the GPU (page cache may be reclaimed)."""
        if hasattr(handler, _MASTERS_ATTR):
            delattr(handler, _MASTERS_ATTR)


def store_from_env(get_env, project_root: str):
    """Build the store selected by ACESTEP_DIT_RESIDENCY_STORE (default pinned)."""
    kind = (get_env(STORE_ENV) or "pinned").strip().lower()
    if kind == "mmap":
        cache_dir = get_env(STORE_DIR_ENV) or os.path.join(project_root, ".cache", "acestep", "dit_residency")
        return MmapStore(cache_dir)
    return PinnedStore()
