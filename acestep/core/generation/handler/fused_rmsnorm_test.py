"""Unit tests for the fused RMSNorm replacement."""

import unittest

import torch
import torch.nn as nn

from acestep.core.generation.handler.fused_rmsnorm import fuse_rmsnorm, fused_rmsnorm_mode


class TinyRMSNorm(nn.Module):  # same math as transformers' Qwen3RMSNorm
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(dtype)


class FusedRMSNormTests(unittest.TestCase):
    def test_mode_parsing(self):
        self.assertIsNone(fused_rmsnorm_mode(None))
        self.assertIsNone(fused_rmsnorm_mode("0"))
        self.assertEqual(fused_rmsnorm_mode("1"), "exact")
        self.assertEqual(fused_rmsnorm_mode("FAST"), "fast")

    def test_fused_matches_reference_and_counts(self):
        torch.manual_seed(0)
        model = nn.Sequential(TinyRMSNorm(64), nn.Linear(64, 64), TinyRMSNorm(64))
        model[0].weight.data.normal_(1.0, 0.1)
        x = torch.randn(3, 10, 64)
        ref = model(x)
        self.assertEqual(fuse_rmsnorm(model, "exact"), 2)
        self.assertEqual(fuse_rmsnorm(model, "exact"), 0)  # idempotent
        out = model(x)
        self.assertTrue(torch.allclose(ref, out, atol=1e-5, rtol=1e-5))
        self.assertEqual(fuse_rmsnorm(model, "fast"), 2)
        self.assertTrue(torch.allclose(ref, model(x), atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
