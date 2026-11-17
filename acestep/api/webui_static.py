"""Serve the HTML/JS evaluation UI (``webui/``) from the API server at ``/ui``.

The UI is a plain static page that talks to this server's REST endpoints
(``/release_task``, ``/query_result``, ``/format_input``, ...). Mounting it on
the same origin avoids CORS setup; the directory is optional so API-only
deployments are unaffected.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

WEBUI_ROUTE = "/ui"


def webui_dir(project_root: str) -> str:
    """Directory holding the static UI files."""
    return os.path.join(project_root, "webui")


def mount_webui(app: Any, project_root: str) -> bool:
    """Mount ``webui/`` at ``/ui`` when the directory exists; return whether it was mounted."""
    directory = webui_dir(project_root)
    if not os.path.isdir(directory):
        return False
    from starlette.staticfiles import StaticFiles

    app.mount(WEBUI_ROUTE, StaticFiles(directory=directory, html=True), name="webui")
    logger.info("[API Server] Web UI mounted at {} from {}", WEBUI_ROUTE, directory)
    return True
