#!/usr/bin/env python
"""Environment check used by setup.sh / setup.bat: torch, CUDA GPUs, flash-attn.

Kept in a file (not `python -c`) because cmd.exe expands `%...%` pairs inside a .bat line and
PowerShell drops inner quotes for native commands, which both mangle one-liners on Windows.
"""
import importlib.util

import torch

ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda available: {ok}")
if ok:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  {p.total_memory / 2**30:.1f} GB  sm_{p.major}{p.minor}")
else:
    print("  no CUDA GPU visible: check the NVIDIA driver (CUDA 12.8 or newer) and CUDA_VISIBLE_DEVICES")
if importlib.util.find_spec("flash_attn") is None:
    print("flash_attn not installed; the server will use SDPA (about 10-25% slower)")
else:
    try:
        import flash_attn

        print(f"flash_attn {flash_attn.__version__} (used by the DiT attention)")
    except Exception as exc:  # noqa: BLE001  the server falls back to SDPA
        print(f"flash_attn not importable ({exc}); the server will use SDPA")
