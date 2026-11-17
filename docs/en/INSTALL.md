# Installation (ACE-Step-1.5-fast)

Target: Linux x86_64 with an NVIDIA GPU (driver supporting CUDA 12.8). Python 3.11 / 3.12 (uv provisions one if missing).

## Requirements

| Item | Guideline |
|---|---|
| GPU | 16 GB (RTX A4000 class) runs the XL models; 40 GB+ can keep every model resident (`ACESTEP_GPU_PROFILE=large`) |
| Host RAM | 32 GB or more (the 16gb profile keeps ~4 GB resident plus page cache) |
| Disk | ~8 GB environment, ~45 GB checkpoints (xl-sft 19 / xl-turbo 19 / 2B turbo 4.5 / LM 0.6B 1.3 / VAE + text encoder 1.5), 24 GB swap cache |
| Network | First run downloads from Hugging Face (ModelScope fallback) |

## Steps

```bash
git clone <this repository> ACE-Step-1.5-fast
cd ACE-Step-1.5-fast
./setup.sh              # uv -> .venv from uv.lock -> GPU check -> checkpoints
./run_api_server_fast.sh
```

Open `http://localhost:8001/ui/` for the evaluation UI; the REST API is on `http://localhost:8001` (see [API.md](API.md)).

- `./setup.sh --large` also fetches the 4B LM (for `ACESTEP_GPU_PROFILE=large` on 40 GB+ GPUs).
- `./setup.sh --no-models` builds the environment only (models are fetched on first start).
- Safe to re-run; existing models are skipped.

## Windows (not yet verified)

```bat
setup.bat
run_api_server_fast.bat
close_api_server.bat 8001
```

Python 3.11 is required (the prebuilt flash-attn wheel for Windows is cp311; uv provisions it). torch is 2.7.1+cu128. Override settings with `set ACESTEP_GPU_PROFILE=large` before launching. `PYTORCH_ALLOC_CONF` is not set on Windows.

## Manual steps

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen                                    # torch 2.10 (cu128), prebuilt flash-attn wheel, everything else
uv run acestep-download --model acestep-v15-xl-sft  # also fetches the main set (VAE, text encoder, 2B turbo, LM 1.7B)
uv run acestep-download --model acestep-v15-xl-turbo
uv run acestep-download --model acestep-5Hz-lm-0.6B
```

Set `ACESTEP_CHECKPOINTS_DIR` in `.env` (template: `.env.example`) to store checkpoints elsewhere.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cuda available: False` | Check the driver supports CUDA 12.8 (`nvidia-smi` shows CUDA Version >= 12.8) |
| `flash_attn not importable` | The server falls back to SDPA (10-25% slower); re-run `uv sync --frozen` |
| OOM at startup | Keep `ACESTEP_GPU_PROFILE=16gb` (default) on 16 GB GPUs and make sure nothing else uses the GPU |
| Generation stops at 600 s | Raise `ACESTEP_GENERATION_TIMEOUT` in `.env` |

Launch settings (profiles, environment variables): [起動コマンド.md](../../起動コマンド.md). Measurements: [docs/ja/低VRAM_XL_プローブ_20261116.md](../ja/低VRAM_XL_プローブ_20261116.md).
