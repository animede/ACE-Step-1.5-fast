"""REST endpoints for exclusive DiT residency: which DiT is on the GPU, and pre-switching.

- ``GET /v1/residency``: mode, the active model and every slot (model name, loaded, on GPU).
- ``POST /v1/residency/activate`` ``{"model": name}`` or ``{"slot": n}``: move that DiT onto
  the GPU now (waits for in-flight jobs), so the next request does not pay the swap.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel


class ActivateRequest(BaseModel):
    model: Optional[str] = None
    slot: Optional[int] = None


def residency_slots(app_state: Any, get_model_name: Callable[[str], str]) -> List[Dict[str, Any]]:
    """Describe the DiT slots (1..3) and whether each one is on the GPU."""
    residency = getattr(app_state, "dit_residency", None)
    slots = []
    for n in (1, 2, 3):
        suffix = "" if n == 1 else str(n)
        handler = getattr(app_state, f"handler{suffix}", None)
        config = getattr(app_state, f"_config_path{suffix}", "") or ""
        if handler is None or not config:
            continue
        loaded = bool(getattr(app_state, f"_initialized{suffix}", False))
        on_gpu = loaded and (residency is None or residency.active is handler)
        slots.append({"slot": n, "model": get_model_name(config), "loaded": loaded, "on_gpu": on_gpu})
    return slots


def _find_handler(app_state: Any, slots: List[Dict[str, Any]], req: ActivateRequest):
    for s in slots:
        if (req.slot and s["slot"] == req.slot) or (req.model and s["model"] == req.model):
            suffix = "" if s["slot"] == 1 else str(s["slot"])
            return getattr(app_state, f"handler{suffix}"), s
    return None, None


def register_residency_routes(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[..., Dict[str, Any]],
    get_model_name: Callable[[str], str],
) -> None:
    """Register the residency status and activate routes on ``app``."""

    @app.get("/v1/residency")
    async def residency_status(_: None = Depends(verify_api_key)):
        residency = getattr(app.state, "dit_residency", None)
        slots = residency_slots(app.state, get_model_name)
        active = next((s["model"] for s in slots if s["on_gpu"]), None)
        return wrap_response({"mode": "exclusive" if residency is not None else "shared",
                              "store": getattr(getattr(residency, "store", None), "name", None),
                              "active": active, "slots": slots})

    @app.post("/v1/residency/activate")
    async def residency_activate(request: ActivateRequest, _: None = Depends(verify_api_key)):
        residency = getattr(app.state, "dit_residency", None)
        if residency is None:
            return wrap_response(None, code=400, error="DiT residency is not exclusive on this server")
        slots = residency_slots(app.state, get_model_name)
        handler, slot = _find_handler(app.state, slots, request)
        if handler is None or not slot["loaded"]:
            return wrap_response(None, code=404, error="No loaded slot matches the request")

        def _switch():
            with residency.exclusive():
                residency.activate(handler)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(app.state.executor, _switch)
        return wrap_response({"active": slot["model"], "slots": residency_slots(app.state, get_model_name)})
