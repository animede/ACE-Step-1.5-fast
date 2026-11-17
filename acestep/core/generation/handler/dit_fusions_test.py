"""CPU unit tests for the DiT fusions (MLP / adaLN math and env parsing)."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from acestep.core.generation.handler.dit_fusions import fuse_mlp, fusions_from_env, _stacked_weight


class Qwen3MLP(nn.Module):  # name matters: fuse_mlp matches on the class name
    def __init__(self, h=16, i=32):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(h, i, bias=False), nn.Linear(h, i, bias=False), nn.Linear(i, h, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DitFusionTests(unittest.TestCase):
    def test_env_parsing(self):
        self.assertEqual(fusions_from_env(None), ())
        self.assertEqual(fusions_from_env("all"), ("qkv", "mlp", "attn", "adaln"))
        self.assertEqual(fusions_from_env("mlp, adaln"), ("mlp", "adaln"))

    def test_fused_mlp_matches_and_shares_storage(self):
        torch.manual_seed(0)
        mlp = Qwen3MLP(); x = torch.randn(2, 5, 16); ref = mlp(x)
        self.assertEqual(fuse_mlp(nn.Sequential(mlp)), 1)
        self.assertTrue(torch.allclose(ref, mlp(x), atol=1e-6))
        self.assertEqual(mlp.gate_proj.weight.data_ptr(), mlp._acestep_gate_up_weight.data_ptr())
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data.clone()  # simulate a device move re-pointing the weight
        self.assertTrue(torch.allclose(ref, mlp(x), atol=1e-6))  # buffer is rebuilt lazily

    def test_stack_weights_keeps_values(self):
        a, b = nn.Linear(4, 3, bias=False), nn.Linear(4, 2, bias=False)
        wa, wb = a.weight.detach().clone(), b.weight.detach().clone()
        holder = nn.Module()
        buf = _stacked_weight(holder, "_buf", [a, b])
        self.assertEqual(tuple(buf.shape), (5, 4))
        self.assertIs(_stacked_weight(holder, "_buf", [a, b]), buf)  # cached while weights alias the buffer
        self.assertTrue(torch.equal(a.weight, wa) and torch.equal(b.weight, wb))


if __name__ == "__main__":
    unittest.main()


class DropStackedWeightsTests(unittest.TestCase):
    def test_drop_then_rebuild(self):
        from acestep.core.generation.handler.dit_fusions import drop_stacked_weights
        torch.manual_seed(0)
        mlp = Qwen3MLP(); x = torch.randn(2, 5, 16); ref = mlp(x)
        fuse_mlp(nn.Sequential(mlp)); mlp(x)
        self.assertEqual(drop_stacked_weights(mlp), 1)
        self.assertIsNone(mlp._acestep_gate_up_weight)
        self.assertTrue(torch.allclose(ref, mlp(x), atol=1e-6))
