"""Unit tests for exclusive DiT residency bookkeeping and host stores (CPU-only)."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from acestep.api.dit_residency import DitResidency, residency_mode_from_env
from acestep.api.dit_residency_store import MmapStore, PinnedStore, store_from_env


def _fake_handler(name: str) -> SimpleNamespace:
    return SimpleNamespace(model=torch.nn.Linear(4, 4), silence_latent=torch.zeros(1, 8, 4), device="cpu",
                           last_init_params={"config_path": f"checkpoints/{name}"})


class DitResidencyTests(unittest.TestCase):
    def test_mode_parsing(self):
        self.assertEqual(residency_mode_from_env(None), "shared")
        self.assertEqual(residency_mode_from_env("Exclusive"), "exclusive")

    def test_activate_parks_previous_and_tracks_active(self):
        a, b = _fake_handler("a"), _fake_handler("b")
        residency = DitResidency(active=a)
        residency.park(b)
        residency.activate(b)
        self.assertIs(residency.active, b)
        residency.activate(b)  # no-op
        residency.activate(a)
        self.assertIs(residency.active, a)

    def test_prime_then_park_reuses_pinned_masters(self):
        a = _fake_handler("a")
        residency = DitResidency(active=a, store=PinnedStore())
        residency.prime(a)
        masters = getattr(a, "_dit_residency_masters")
        self.assertEqual(len(masters), 2)  # weight + bias
        residency.park(a)
        self.assertEqual(a.model.weight.data.data_ptr(), masters["weight"].data_ptr())
        self.assertIsNone(residency.active)

    def test_park_handles_handler_without_model(self):
        DitResidency().park(SimpleNamespace(model=None))


class MmapStoreTests(unittest.TestCase):
    def test_roundtrip_through_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MmapStore(tmp)
            a = _fake_handler("acestep-v15-xl-sft")
            weight = a.model.weight.detach().clone()
            residency = DitResidency(active=a, store=store)
            residency.prime(a)
            path = os.path.join(tmp, "acestep-v15-xl-sft.residency.safetensors")
            self.assertTrue(os.path.exists(path))
            self.assertEqual(store.prepare(a, a.model), 0)  # matching file is reused
            residency.park(a)
            self.assertTrue(torch.equal(a.model.weight.data, weight))
            residency.activate(a)
            self.assertFalse(hasattr(a, "_dit_residency_masters"))  # mapping released
            self.assertTrue(torch.equal(a.model.weight.data, weight))

    def test_store_from_env(self):
        self.assertIsInstance(store_from_env(lambda k: None, "/p"), PinnedStore)
        store = store_from_env(lambda k: {"ACESTEP_DIT_RESIDENCY_STORE": "mmap"}.get(k), "/p")
        self.assertIsInstance(store, MmapStore)
        self.assertEqual(store.cache_dir, "/p/.cache/acestep/dit_residency")


if __name__ == "__main__":
    unittest.main()
