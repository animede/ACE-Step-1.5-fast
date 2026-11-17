#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
	# shellcheck disable=SC1091
	source "$ROOT_DIR/.venv/bin/activate"
fi

# -----------------------------------------------------------------------------
# ACE-Step-1.5-fast API server for a 16GB GPU (RTX A4000)
#
#   always loaded : acestep-v15-xl-sft   (primary, bf16, no quantization)
#   choice 1      : acestep-v15-xl-turbo (request {"model": "acestep-v15-xl-turbo"})
#   choice 2      : acestep-v15-turbo    (request {"model": "acestep-v15-turbo"}, 2B)
#
# All three are read from disk once at startup. Only ONE DiT lives on the GPU;
# the others wait in pinned host memory (~23 GB RAM for the three) and are
# swapped in over PCIe in about 2 s when requested. The primary returns to the
# GPU on the next request for it. Fused RMSNorm, resident VAE/LM and 512-frame
# VAE decode are on (A4000, XL-SFT 300 s song: ~62-65 s).
# -----------------------------------------------------------------------------

export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-xl-sft}"
export ACESTEP_CONFIG_PATH2="${ACESTEP_CONFIG_PATH2:-acestep-v15-xl-turbo}"
export ACESTEP_CONFIG_PATH3="${ACESTEP_CONFIG_PATH3:-acestep-v15-turbo}"

export ACESTEP_DIT_RESIDENCY="${ACESTEP_DIT_RESIDENCY:-exclusive}"
export ACESTEP_QUANTIZATION="${ACESTEP_QUANTIZATION:-none}"
export ACESTEP_COMPILE_MODEL="${ACESTEP_COMPILE_MODEL:-0}"
export ACESTEP_FUSED_RMSNORM="${ACESTEP_FUSED_RMSNORM:-exact}"

export ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-1}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-0.6B}"
export ACESTEP_LM_BACKEND="${ACESTEP_LM_BACKEND:-pt}"

export ACESTEP_DEVICE="${ACESTEP_DEVICE:-auto}"
export ACESTEP_OFFLOAD_TO_CPU="${ACESTEP_OFFLOAD_TO_CPU:-1}"
export ACESTEP_RESIDENT_COMPONENTS="${ACESTEP_RESIDENT_COMPONENTS:-vae}"
export ACESTEP_LM_OFFLOAD_TO_CPU="${ACESTEP_LM_OFFLOAD_TO_CPU:-0}"
export ACESTEP_OFFLOAD_DIT_TO_CPU="${ACESTEP_OFFLOAD_DIT_TO_CPU:-0}"
export ACESTEP_VAE_DECODE_CHUNK_SIZE="${ACESTEP_VAE_DECODE_CHUNK_SIZE:-512}"
# export ACESTEP_MP3_COMPRESSION_LEVEL=7   # optional LAME speed preset (-1.7 s / 300 s song)
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
