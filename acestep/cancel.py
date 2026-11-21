"""Cooperative cancellation of a running generation.

The API worker runs one generation at a time in an executor thread.  Before
calling into the model it binds the job's cancel event to that thread
(``bind``); the diffusion loops in ``acestep/models/*/modeling_*.py`` call
``check()`` once per step and raise ``GenerationCancelled`` when the event is
set.  ``POST /v1/cancel`` sets the event (and marks a still-queued job so the
worker skips it).  Nothing is torn down forcibly: the step finishes, the
exception unwinds through the handler, and the job is marked ``cancelled``.
"""

from __future__ import annotations

import threading
from typing import Optional

_local = threading.local()


class GenerationCancelled(RuntimeError):
    """Raised inside the diffusion loop when the job was cancelled by a client."""


def bind(event: Optional[threading.Event]) -> None:
    """Attach *event* to the current thread (``None`` detaches)."""
    _local.event = event


def current() -> Optional[threading.Event]:
    return getattr(_local, "event", None)


def check() -> None:
    """Call once per diffusion step; raises ``GenerationCancelled`` if a cancel was requested."""
    ev = getattr(_local, "event", None)
    if ev is not None and ev.is_set():
        raise GenerationCancelled("cancelled by client")
