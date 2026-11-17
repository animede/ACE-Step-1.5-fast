"""Tests for legacy NVIDIA torch compatibility launcher guards."""

from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LauncherLegacyTorchFixTests(unittest.TestCase):
    """Ensure launcher compatibility fixes are narrowly scoped and opt-out capable."""

    def _read(self, rel_path: str) -> str:
        """Read a launcher script from the project root."""
        return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")

    def test_linux_api_launcher_calls_shared_probe(self) -> None:
        """Linux API launcher should call shared Python compatibility probe."""
        content = self._read("start_api_server.sh")
        self.assertIn("ACESTEP_SKIP_LEGACY_TORCH_FIX", content)
        self.assertIn("legacy_torch_fix_probe_exit_code", content)
        self.assertIn("legacy NVIDIA compatibility probe failed with exit code $compat_status", content)
        self.assertIn("return 1", content)
        self.assertIn("torch==2.5.1+cu121", content)

