"""Unit tests for mounting the static web UI."""

import os
import tempfile
import unittest

from fastapi import FastAPI

from acestep.api.webui_static import mount_webui


class WebuiMountTests(unittest.TestCase):
    def test_mounts_when_directory_exists(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "webui"))
            with open(os.path.join(root, "webui", "index.html"), "w") as f:
                f.write("<html></html>")
            app = FastAPI()
            self.assertTrue(mount_webui(app, root))
            self.assertTrue(any(getattr(r, "path", "") == "/ui" for r in app.routes))

    def test_skips_without_directory(self):
        with tempfile.TemporaryDirectory() as root:
            app = FastAPI()
            self.assertFalse(mount_webui(app, root))
            self.assertFalse(any(getattr(r, "path", "") == "/ui" for r in app.routes))


if __name__ == "__main__":
    unittest.main()
