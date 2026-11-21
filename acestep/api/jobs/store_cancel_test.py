"""Cancel behaviour of the in-memory job store and the cooperative check."""

from __future__ import annotations

import threading
import unittest

from acestep import cancel
from acestep.api.jobs.store import _JobStore


class StoreCancelTests(unittest.TestCase):
    def test_queued_job_is_cancelled_immediately(self) -> None:
        store = _JobStore()
        rec = store.create()
        self.assertEqual("cancelled", store.request_cancel(rec.job_id))
        self.assertEqual("cancelled", store.get(rec.job_id).status)
        self.assertTrue(store.cancel_requested(rec.job_id))
        self.assertEqual("cancelled by client", store.get(rec.job_id).error)

    def test_running_job_gets_cancelling_and_event(self) -> None:
        store = _JobStore()
        rec = store.create()
        store.mark_running(rec.job_id)
        ev = store.cancel_event(rec.job_id)
        self.assertFalse(ev.is_set())
        self.assertEqual("cancelling", store.request_cancel(rec.job_id))
        self.assertTrue(ev.is_set())
        self.assertEqual("running", store.get(rec.job_id).status)  # the worker marks it after the step
        store.mark_cancelled(rec.job_id)
        self.assertEqual("cancelled", store.get(rec.job_id).status)

    def test_finished_and_unknown(self) -> None:
        store = _JobStore()
        rec = store.create()
        store.mark_running(rec.job_id)
        store.mark_succeeded(rec.job_id, {"ok": True})
        self.assertEqual("succeeded", store.request_cancel(rec.job_id))
        self.assertIsNone(store.request_cancel("nope"))

    def test_check_raises_only_on_bound_set_event(self) -> None:
        cancel.bind(None)
        cancel.check()  # no event: no-op
        ev = threading.Event()
        cancel.bind(ev)
        cancel.check()
        ev.set()
        with self.assertRaises(cancel.GenerationCancelled):
            cancel.check()
        cancel.bind(None)
        cancel.check()

    def test_cancelled_counts_in_stats_and_cleanup(self) -> None:
        store = _JobStore(max_age_seconds=0)
        rec = store.create()
        store.request_cancel(rec.job_id)
        self.assertEqual(1, store.get_stats()["cancelled"])
        store.get(rec.job_id).finished_at -= 10
        self.assertEqual(1, store.cleanup_old_jobs())


if __name__ == "__main__":
    unittest.main()
