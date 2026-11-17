"""Unit tests for the generation timing summary."""

import unittest

from acestep.api.generation_info import build_generation_info


class GenerationInfoTests(unittest.TestCase):
    def test_summary_lines(self):
        text = build_generation_info(None, {"lm_total_time": 2.0, "dit_total_time_cost": 8.0, "audio_conversion_time": 1.5}, "1", 8, 2, "mp3")
        self.assertIn("Total generation time (2 songs): 10.00s", text)
        self.assertIn("5.00s per song", text)
        self.assertIn("to MP3 (2 songs): 1.50s", text)

    def test_empty_when_nothing_to_report(self):
        self.assertEqual(build_generation_info(None, {}, "1", 8, 1), "")
        self.assertEqual(build_generation_info(None, {"dit_total_time_cost": 1.0}, "1", 8, 0), "")


if __name__ == "__main__":
    unittest.main()
