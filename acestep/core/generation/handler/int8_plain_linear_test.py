"""Unit tests for the plain INT8 weight-only Linear."""

import unittest

import torch
import torch.nn as nn

from acestep.core.generation.handler.int8_plain_linear import Int8PlainLinear, quantize_linears


class Int8PlainLinearTests(unittest.TestCase):
    def test_forward_matches_bf16_linear_closely(self):
        torch.manual_seed(0)
        lin = nn.Linear(64, 32, dtype=torch.bfloat16)
        q = Int8PlainLinear(lin)
        x = torch.randn(8, 64, dtype=torch.bfloat16)
        ref, out = lin(x).float(), q(x).float()
        self.assertEqual(q.weight_int8.dtype, torch.int8)
        self.assertLess((ref - out).abs().max().item() / ref.abs().max().item(), 0.05)

    def test_quantize_linears_only_touches_decoder_outside_tokenizers(self):
        model = nn.Module()
        model.decoder = nn.Module()
        model.decoder.proj = nn.Linear(4, 4)
        model.decoder.tokenizer = nn.Module()
        model.decoder.tokenizer.proj = nn.Linear(4, 4)
        model.encoder = nn.Module()
        model.encoder.proj = nn.Linear(4, 4)
        self.assertEqual(quantize_linears(model), 1)
        self.assertIsInstance(model.decoder.proj, Int8PlainLinear)
        self.assertIsInstance(model.decoder.tokenizer.proj, nn.Linear)
        self.assertIsInstance(model.encoder.proj, nn.Linear)


if __name__ == "__main__":
    unittest.main()
