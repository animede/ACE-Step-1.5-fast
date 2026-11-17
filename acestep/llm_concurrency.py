"""Serialise access to the 5Hz LM handler across generation threads.

The PyTorch LM backend keeps per-call state on the handler (inputs moved to the
device, optional CPU offload of the model), so two queue workers calling it at
the same time trigger CUDA device-side asserts that poison the whole process.
The decorator below gives each handler one re-entrant lock; DiT generation is
not affected and still runs in parallel.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable

_LOCK_ATTR = "_acestep_lm_lock"
_LOCK_GUARD = threading.Lock()


def _lock_for(instance: Any) -> threading.RLock:
    lock = getattr(instance, _LOCK_ATTR, None)
    if lock is None:
        with _LOCK_GUARD:
            lock = getattr(instance, _LOCK_ATTR, None)
            if lock is None:
                lock = threading.RLock()
                setattr(instance, _LOCK_ATTR, lock)
    return lock


def serialized(method: Callable) -> Callable:
    """Run ``method`` under the instance's re-entrant lock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with _lock_for(self):
            return method(self, *args, **kwargs)

    return wrapper
