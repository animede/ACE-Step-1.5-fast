"""Unit tests for the per-model VRAM gate."""

from __future__ import annotations

import unittest

from acestep.api.model_vram import min_vram_gb, vram_fits, vram_problem


class ModelVramTests(unittest.TestCase):
    def test_xl_needs_14gb_bf16_and_9gb_int8(self) -> None:
        self.assertEqual(14.0, min_vram_gb("acestep-v15-xl-sft"))
        self.assertEqual(9.0, min_vram_gb("acestep-v15-xl-turbo", "int8_weight_only"))

    def test_2b_fits_a_12gb_card_reported_as_11_9(self) -> None:
        self.assertTrue(vram_fits("acestep-v15-turbo", 11.9))
        self.assertTrue(vram_fits("acestep-v15-base", 7.9, "int8_weight_only"))
        self.assertFalse(vram_fits("acestep-v15-turbo", 4.0))

    def test_xl_refused_on_12gb_with_next_step(self) -> None:
        msg = vram_problem("acestep-v15-xl-sft", 11.9)
        self.assertIsNotNone(msg)
        self.assertTrue(msg.startswith("VRAM too small"))
        self.assertIn("ACESTEP_GPU_PROFILE=12gb", msg)
        self.assertIsNone(vram_problem("acestep-v15-xl-sft", 15.9))

    def test_unknown_gpu_size_does_not_block(self) -> None:
        self.assertTrue(vram_fits("acestep-v15-xl-sft", 0.0))
        self.assertIsNone(vram_problem("acestep-v15-xl-sft", 0.0))


if __name__ == "__main__":
    unittest.main()
