"""Unit tests for exclusive DiT residency bookkeeping (CPU-only)."""

import unittest
from types import SimpleNamespace

import torch

from acestep.api.dit_residency import DitResidency, residency_mode_from_env


def _fake_handler(name: str) -> SimpleNamespace:
    handler = SimpleNamespace(model=torch.nn.Linear(4, 4), silence_latent=torch.zeros(1, 8, 4), device="cpu", model_name=name)
    handler._recursive_to_device = lambda model, device, dtype=None: model.to(device)
    return handler


class DitResidencyTests(unittest.TestCase):
    def test_mode_parsing(self):
        self.assertEqual(residency_mode_from_env(None), "shared")
        self.assertEqual(residency_mode_from_env("Exclusive"), "exclusive")
        self.assertEqual(residency_mode_from_env("shared"), "shared")

    def test_activate_parks_previous_and_tracks_active(self):
        a, b = _fake_handler("a"), _fake_handler("b")
        residency = DitResidency(active=a)
        residency.park(b)
        self.assertIs(residency.active, a)
        residency.activate(b)
        self.assertIs(residency.active, b)
        residency.activate(b)  # no-op
        self.assertIs(residency.active, b)
        residency.activate(a)
        self.assertIs(residency.active, a)
        self.assertEqual(a.silence_latent.device.type, "cpu")

    def test_prime_then_park_makes_no_new_copies(self):
        a = _fake_handler("a")
        residency = DitResidency(active=a)
        residency.prime(a)
        masters = getattr(a, "_dit_residency_masters")
        self.assertEqual(len(masters), 2)  # weight + bias
        residency.park(a)
        self.assertEqual(a.model.weight.data.data_ptr(), masters["weight"].data_ptr())
        self.assertIsNone(residency.active)

    def test_park_handles_handler_without_model(self):
        residency = DitResidency()
        residency.park(SimpleNamespace(model=None))
        self.assertIsNone(residency.active)


if __name__ == "__main__":
    unittest.main()
