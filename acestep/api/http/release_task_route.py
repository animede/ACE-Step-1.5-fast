"""HTTP route registration for ``/release_task`` task-submission endpoint."""

from __future__ import annotations

import asyncio
import uuid
import os
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request

from acestep.api.http.release_task_request_parser import parse_release_task_request


async def _client_task_id(request: Request) -> Optional[str]:
    """Body (JSON or form, already cached by the parser) の task_id を UUID として読む。無効・無しは None。"""
    content_type = (request.headers.get("content-type") or "").lower()
    raw = None
    try:
        if "json" in content_type:
            body = await request.json()
            raw = body.get("task_id") if isinstance(body, dict) else None
        elif "form" in content_type:
            raw = (await request.form()).get("task_id")
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        return str(uuid.UUID(str(raw)))
    except ValueError:
        raise HTTPException(status_code=400, detail="task_id must be a UUID")


def register_release_task_route(
    app: FastAPI,
    verify_token_from_request: Callable[[dict, Optional[str]], Optional[str]],
    wrap_response: Callable[..., Dict[str, Any]],
    store: Any,
    request_parser_cls: Any,
    request_model_cls: Any,
    validate_audio_path: Callable[[Optional[str]], Optional[str]],
    save_upload_to_temp: Callable[..., Any],
    upload_file_type: type,
    default_dit_instruction: str,
    lm_default_temperature: float,
    lm_default_cfg_scale: float,
    lm_default_top_p: float,
) -> None:
    """Register the legacy-compatible ``/release_task`` route.

    Args:
        app: FastAPI app instance.
        verify_token_from_request: Legacy token validator.
        wrap_response: API response wrapper helper.
        store: Job store exposing ``create()``.
        request_parser_cls: Request parser class for body dictionaries.
        request_model_cls: Request model class (for example ``GenerateMusicRequest``).
        validate_audio_path: Validator for manual audio-path fields.
        save_upload_to_temp: Helper for persisting uploaded files to temp paths.
        upload_file_type: Upload class used for multipart file detection.
        default_dit_instruction: Default DiT instruction string.
        lm_default_temperature: Default LM temperature value.
        lm_default_cfg_scale: Default LM CFG scale value.
        lm_default_top_p: Default LM top-p value.
    """

    @app.post("/release_task")
    async def create_music_generate_job(request: Request, authorization: Optional[str] = Header(None)):
        """Create a queued generation job from supported content types."""

        req, temp_files = await parse_release_task_request(
            request=request,
            authorization=authorization,
            verify_token_from_request=verify_token_from_request,
            request_parser_cls=request_parser_cls,
            request_model_cls=request_model_cls,
            validate_audio_path=validate_audio_path,
            save_upload_to_temp=save_upload_to_temp,
            upload_file_type=upload_file_type,
            default_dit_instruction=default_dit_instruction,
            lm_default_temperature=lm_default_temperature,
            lm_default_cfg_scale=lm_default_cfg_scale,
            lm_default_top_p=lm_default_top_p,
        )

        # 任意: クライアントが task_id を決めて送れる(UUID 形式)。投入の応答が遅れてクライアントが待ちを
        # やめても、その ID で /v1/cancel できる(応答を受け取れないと ID が分からず、曲が宙に浮くため)。
        client_task_id = await _client_task_id(request)
        if client_task_id and store.get(client_task_id) is not None:
            raise HTTPException(status_code=409, detail=f"task_id {client_task_id} already exists")
        record = store.create_with_id(client_task_id) if client_task_id else store.create()
        queue_ref: asyncio.Queue = app.state.job_queue
        if queue_ref.full():
            for temp_path in temp_files:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise HTTPException(status_code=429, detail="Server busy: queue is full")

        if temp_files:
            async with app.state.job_temp_files_lock:
                app.state.job_temp_files[record.job_id] = temp_files

        async with app.state.pending_lock:
            app.state.pending_ids.append(record.job_id)
            position = len(app.state.pending_ids)

        await queue_ref.put((record.job_id, req))
        return wrap_response({"task_id": record.job_id, "status": "queued", "queue_position": position})
