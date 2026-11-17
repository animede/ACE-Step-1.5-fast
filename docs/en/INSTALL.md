# Installation (ACE-Step-1.5-fast)

Target: Linux x86_64 with an NVIDIA GPU (driver supporting CUDA 12.8). Python 3.11 / 3.12 (uv provisions one if missing).

## Requirements

| Item | Guideline |
|---|---|
| GPU | 16 GB (RTX A4000 class) runs the XL models; 40 GB+ can keep every model resident (`ACESTEP_GPU_PROFILE=large`); 12 GB (RTX 3060 12 GB etc.) runs the 2B models only (`ACESTEP_GPU_PROFILE=12gb`, peak about 8.5 GB) |
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
- `./setup.sh --12gb` skips the XL models and fetches 2B turbo + 2B base + LM 0.6B only (about 15 GB with the main set's 1.7B LM, for `ACESTEP_GPU_PROFILE=12gb` on 12 GB GPUs).
- `./setup.sh --no-models` builds the environment only (models are fetched on first start).
- Safe to re-run; existing models are skipped.

## Windows

Prerequisites: Windows 10 / 11 (64-bit), an NVIDIA driver supporting CUDA 12.8, 32 GB RAM, 80 GB disk. No Python, CUDA Toolkit or Visual Studio needed.
Run these in a command prompt, top to bottom:

```bat
winget install --id Git.Git -e
git clone https://github.com/animede/ACE-Step-1.5-fast.git
cd ACE-Step-1.5-fast
setup.bat
run_api_server_fast.bat
```

```bat
close_api_server.bat
```

- Then open `http://localhost:8001/ui/`. Stop another port with `close_api_server.bat 8002`.
- `setup.bat`: installs uv (winget when available: `winget install --id astral-sh.uv -e`, otherwise the official PowerShell installer), provisions Python 3.11 and `.venv` (`uv sync --frozen`), checks the GPU, downloads the models. `setup.bat --large` also fetches the 4B LM, `setup.bat --12gb` fetches the 2B models only, `setup.bat --no-models` builds the environment only. Safe to re-run; if `uv` is not found right after installing, open a new command prompt and run it again.
- Without Git: download the ZIP from GitHub ("Code -> Download ZIP"), extract it and continue from `setup.bat` in that folder.
- Override settings before launching, e.g. `set ACESTEP_GPU_PROFILE=large` (same variables as 起動コマンド.md). `PYTORCH_ALLOC_CONF` is not set on Windows.
- The Windows scripts have not been exercised on a real machine yet; please open an issue if something fails.

## Running the bundled Easy Music app and choosing the LLM

`./setup.sh` / `setup.bat` also build the environment for Easy Music (`easy-music/`). With the API server running, start it in another terminal (the argument is the port):

```bash
cd ACE-Step-1.5-fast
./run_easy_music.sh 8888
```

On Windows: `run_easy_music.bat 8888`. Then open `http://localhost:8888`.

Lyrics are written by an OpenAI-compatible LLM API (Ollama, llama.cpp server, LM Studio, vLLM, OpenAI, ...); the default is `http://localhost:11434/v1` (Ollama). If it cannot be reached, the app falls back to a CPU GGUF model (Gemma 4 E4B Q4_0, about 4.4GB, downloaded on first use, thinking off). The bundled LLM is llama.cpp's prebuilt `llama-server` (CPU build, about 25MB) from the GitHub Releases: it is fetched into `easy-music/tools/llama/` on the first fallback and run as a child process. No compiler or Build Tools needed (`easy-music/scripts/fetch_llama_server.py` pre-fetches it; point `LLAMA_SERVER` / `LOCAL_LLM_GGUF` at existing files). Two ways to set it, in order of precedence: launch arguments > `easy-music/.env` > defaults.

Launch arguments:

```bash
./run_easy_music.sh 8888 --llm-url http://LLM_HOST:11434/v1 --llm-model gemma3:latest
```

(or `--llm-host LLM_HOST --llm-port 11434`; on Windows `run_easy_music.bat 8888 --llm-url ...`)

`easy-music/.env` (persistent; keep `OPENAI_API_KEY` as a dummy for servers without authentication):

```
OPENAI_BASE_URL=http://LLM_HOST:11434/v1
OPENAI_API_KEY=YOUR_KEY
OPENAI_CHAT_MODEL=gemma3:latest
```

In the same way, the ACE server is set with `--ace-url` / `ACE_STEP_API_URL` (default `http://127.0.0.1:8001`) and the optional background-image server with `--img-url` / `IMAGE_GENERATION_URL`.

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
