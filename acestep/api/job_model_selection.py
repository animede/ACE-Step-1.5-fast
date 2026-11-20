"""Model selection helpers for per-job DiT handler routing."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple


class ModelNotLoadedError(ValueError):
    """The request named a DiT model that is not in any loaded slot."""

    def __init__(self, requested: str, loaded: list):
        self.requested = requested
        self.loaded = list(loaded)
        super().__init__(
            f"Model '{requested}' is not loaded on this server (loaded: {', '.join(self.loaded)}). "
            f"Next step: choose one of the loaded models, or load it first with POST /v1/init "
            f"{{\"model\": \"{requested}\", \"slot\": 2}} (Easy Music does this when you pick it)."
        )


def select_generation_handler(
    *,
    app_state: Any,
    requested_model: Optional[str],
    get_model_name: Callable[[str], str],
    job_id: str,
    log_fn: Callable[[str], None] = print,
) -> Tuple[Any, str]:
    """Resolve the handler/model name for a generation job request.

    Args:
        app_state: Application state object containing primary/secondary/third handlers and config paths.
        requested_model: Optional requested model name from request payload.
        get_model_name: Callable that normalizes config path to display model name.
        job_id: Current job identifier for log messages.
        log_fn: Logger callable used for parity with existing print-based logs.

    Returns:
        Tuple of ``(selected_handler, selected_model_name)``.
    """

    selected_handler = app_state.handler
    selected_model_name = get_model_name(app_state._config_path)

    if not requested_model:
        return selected_handler, selected_model_name

    model_matched = False

    if app_state.handler2 and getattr(app_state, "_initialized2", False):
        model2_name = get_model_name(app_state._config_path2)
        if requested_model == model2_name:
            selected_handler = app_state.handler2
            selected_model_name = model2_name
            model_matched = True
            log_fn(f"[API Server] Job {job_id}: Using second model: {model2_name}")

    if not model_matched and app_state.handler3 and getattr(app_state, "_initialized3", False):
        model3_name = get_model_name(app_state._config_path3)
        if requested_model == model3_name:
            selected_handler = app_state.handler3
            selected_model_name = model3_name
            model_matched = True
            log_fn(f"[API Server] Job {job_id}: Using third model: {model3_name}")

    if not model_matched and requested_model != selected_model_name:
        available_models = [selected_model_name]
        if app_state.handler2 and getattr(app_state, "_initialized2", False):
            available_models.append(get_model_name(app_state._config_path2))
        if app_state.handler3 and getattr(app_state, "_initialized3", False):
            available_models.append(get_model_name(app_state._config_path3))
        # Refuse instead of quietly generating with the primary model: a user who
        # picked "turbo" and silently got XL-SFT only sees a slow, wrong result.
        log_fn(
            f"[API Server] Job {job_id}: Model '{requested_model}' not loaded "
            f"(loaded: {available_models}); refusing"
        )
        raise ModelNotLoadedError(requested_model, available_models)

    return selected_handler, selected_model_name
