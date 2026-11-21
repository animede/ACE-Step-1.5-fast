"""``POST /v1/cancel``: stop a queued or running generation job."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field


class CancelRequest(BaseModel):
    task_id: str = Field(..., description="task_id returned by /release_task")


def register_cancel_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[..., Dict[str, Any]],
    store: Any,
) -> None:
    """Register the cancel route.

    A queued job is marked cancelled at once and skipped by the worker. A running
    job gets its cancel event set; the diffusion loop checks it once per step and
    stops after the current one (a few seconds at most), so the GPU is free for the
    next request without waiting for the whole song. Both end up as status 2
    (failed) in /query_result with error "cancelled by client".
    """

    @app.post("/v1/cancel")
    async def cancel_task(request: CancelRequest, _: None = Depends(verify_api_key)):
        state: Optional[str] = store.request_cancel(request.task_id)
        if state is None:
            return wrap_response(None, code=404, error=f"unknown task_id {request.task_id}")
        print(f"[API Server] Job {request.task_id}: cancel requested -> {state}")
        return wrap_response({"task_id": request.task_id, "status": state})
