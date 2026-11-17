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


class ResidencyConcurrencyTests(unittest.TestCase):
    def test_switch_waits_for_inflight_jobs_and_same_model_runs_in_parallel(self):
        import threading
        import time as _time

        a, b = _fake_handler("a"), _fake_handler("b")
        residency = DitResidency(active=a)
        residency.park(b)
        events = []
        a_started, release_a = threading.Event(), threading.Event()

        def job_a(tag):
            with residency.use(a):
                events.append(f"{tag}:start"); a_started.set(); release_a.wait(5); events.append(f"{tag}:end")

        def job_b():
            with residency.use(b):
                events.append("b:start"); events.append("b:end")

        t1 = threading.Thread(target=job_a, args=("a1",)); t1.start(); a_started.wait(5)
        t2 = threading.Thread(target=job_a, args=("a2",)); t2.start(); _time.sleep(0.2)  # same model: runs alongside a1
        t3 = threading.Thread(target=job_b); t3.start(); _time.sleep(0.2)
        self.assertNotIn("b:start", events)  # b waits for a1/a2
        self.assertEqual(residency._switch_waiting, 1)
        release_a.set(); t1.join(5); t2.join(5); t3.join(5)
        self.assertEqual(events[-2:], ["b:start", "b:end"])
        self.assertIn("a2:start", events[:3])
        self.assertIs(residency.active, b)
        self.assertEqual(residency._in_use, 0)
