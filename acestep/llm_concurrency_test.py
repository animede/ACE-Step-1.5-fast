"""Unit tests for the LM serialisation decorator."""

import threading
import time
import unittest

from acestep.llm_concurrency import serialized


class _Handler:
    def __init__(self):
        self.active = 0
        self.max_active = 0

    @serialized
    def generate(self, delay=0.05):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep(delay)
        self.active -= 1
        return "ok"

    @serialized
    def outer(self):
        return self.generate(0.0)  # re-entrant


class SerializedTests(unittest.TestCase):
    def test_calls_never_overlap(self):
        h = _Handler()
        threads = [threading.Thread(target=h.generate) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertEqual(h.max_active, 1)

    def test_reentrant(self):
        self.assertEqual(_Handler().outer(), "ok")


if __name__ == "__main__":
    unittest.main()
