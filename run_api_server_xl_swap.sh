#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
	# shellcheck disable=SC1091
	source "$ROOT_DIR/.venv/bin/activate"
fi

# -----------------------------------------------------------------------------
# XL-SFT resident, XL-turbo swapped in on demand (16GB GPU, measured on RTX A4000)
#
# - Only ONE DiT lives on the GPU. The primary (xl-sft, bf16, no quantization)
#   starts on the GPU; the secondary (xl-turbo) is loaded at startup but parked
#   in pinned host memory (no VRAM). A request for the other model swaps them
#   over PCIe (about 1 s), then the active model has the whole GPU.
# - Both DiTs stay bf16: the original quality, no torchao.
# - LM 0.6B (pt backend), VAE / text encoder offloaded between uses.
#
# Per-request model switch: {"model": "acestep-v15-xl-turbo"} in /release_task.
# -----------------------------------------------------------------------------

export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-xl-sft}"
export ACESTEP_CONFIG_PATH2="${ACESTEP_CONFIG_PATH2:-acestep-v15-xl-turbo}"
export ACESTEP_CONFIG_PATH3="${ACESTEP_CONFIG_PATH3:-}"

export ACESTEP_DIT_RESIDENCY="${ACESTEP_DIT_RESIDENCY:-exclusive}"
export ACESTEP_QUANTIZATION="${ACESTEP_QUANTIZATION:-none}"
# Fused RMSNorm in the DiT (one CUDA kernel instead of ~7 per norm): 1 step -14%,
# same math (bf16 last-bit rounding only). "exact" keeps Qwen3's rounding order.
export ACESTEP_FUSED_RMSNORM="${ACESTEP_FUSED_RMSNORM:-exact}"
export ACESTEP_COMPILE_MODEL="${ACESTEP_COMPILE_MODEL:-0}"

export ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-1}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-0.6B}"
export ACESTEP_LM_BACKEND="${ACESTEP_LM_BACKEND:-pt}"

export ACESTEP_DEVICE="${ACESTEP_DEVICE:-auto}"
# Text encoder is offloaded between uses; the VAE (0.3 GB) and the LM (1.2 GB)
# stay on the GPU: saves ~3.5 s of CPU<->GPU round trips per job (16 GB budget:
# sft bf16 9.2 + VAE/LM 1.5 + CUDA context, peak 13.6 GB for a 300 s song).
export ACESTEP_OFFLOAD_TO_CPU="${ACESTEP_OFFLOAD_TO_CPU:-1}"
export ACESTEP_RESIDENT_COMPONENTS="${ACESTEP_RESIDENT_COMPONENTS:-vae}"
export ACESTEP_LM_OFFLOAD_TO_CPU="${ACESTEP_LM_OFFLOAD_TO_CPU:-0}"
# VAE decode in 512-frame tiles instead of the auto 128 (-2 s for 300 s, +0.5 GB peak).
export ACESTEP_VAE_DECODE_CHUNK_SIZE="${ACESTEP_VAE_DECODE_CHUNK_SIZE:-512}"
# Optional: LAME speed preset (same bitrate, -1.7 s for 300 s). Off by default; A/B first.
# export ACESTEP_MP3_COMPRESSION_LEVEL=7
export ACESTEP_OFFLOAD_DIT_TO_CPU="${ACESTEP_OFFLOAD_DIT_TO_CPU:-0}"
export ACESTEP_NO_INIT="${ACESTEP_NO_INIT:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

export ACESTEP_API_WORKERS="${ACESTEP_API_WORKERS:-1}"
HOST="${ACESTEP_API_HOST:-0.0.0.0}"
PORT="${ACESTEP_API_PORT:-8001}"
LOG_LEVEL="${ACESTEP_API_LOG_LEVEL:-info}"

exec python -m uvicorn acestep.api_server:app \
	--host "$HOST" \
	--port "$PORT" \
	--workers 1 \
	--log-level "$LOG_LEVEL"
