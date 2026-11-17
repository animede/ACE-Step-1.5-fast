#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
	# shellcheck disable=SC1091
	source "$ROOT_DIR/.venv/bin/activate"
fi

# -----------------------------------------------------------------------------
# ACE-Step-1.5-fast API server (the only launcher of this fork)
#
#   always loaded : acestep-v15-xl-sft   (primary, bf16, no quantization)
#   choice 1      : acestep-v15-xl-turbo (request {"model": "acestep-v15-xl-turbo"})
#   choice 2      : acestep-v15-turbo    (request {"model": "acestep-v15-turbo"}, 2B)
#
# ACESTEP_GPU_PROFILE selects the memory strategy:
#   16gb  (default) one DiT on the GPU at a time; the others wait in the mmap
#         store (bf16 safetensors in the page cache, ~24 GB disk, server RSS ~4 GB)
#         and are swapped in over PCIe in about 2 s. VAE + LM 0.6B resident,
#         text encoder offloaded. Measured on RTX A4000: XL-SFT 300 s song 62-65 s.
#   large (>= 40 GB VRAM) every DiT stays on the GPU (no swap), nothing is
#         offloaded, LM 4B. XL-SFT + XL-turbo + 2B turbo + LM 4B ~ 33 GB.
# Every value below can be overridden from the environment.
# -----------------------------------------------------------------------------

export ACESTEP_GPU_PROFILE="${ACESTEP_GPU_PROFILE:-16gb}"
if [[ "$ACESTEP_GPU_PROFILE" == "large" ]]; then
	export ACESTEP_DIT_RESIDENCY="${ACESTEP_DIT_RESIDENCY:-shared}"
	export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-4B}"
	export ACESTEP_OFFLOAD_TO_CPU="${ACESTEP_OFFLOAD_TO_CPU:-0}"
	export ACESTEP_LM_OFFLOAD_TO_CPU="${ACESTEP_LM_OFFLOAD_TO_CPU:-0}"
	export ACESTEP_RESIDENT_COMPONENTS="${ACESTEP_RESIDENT_COMPONENTS:-}"
	export ACESTEP_VAE_DECODE_CHUNK_SIZE="${ACESTEP_VAE_DECODE_CHUNK_SIZE:-1024}"
fi

export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-xl-sft}"
export ACESTEP_CONFIG_PATH2="${ACESTEP_CONFIG_PATH2:-acestep-v15-xl-turbo}"
export ACESTEP_CONFIG_PATH3="${ACESTEP_CONFIG_PATH3:-acestep-v15-turbo}"

export ACESTEP_DIT_RESIDENCY="${ACESTEP_DIT_RESIDENCY:-exclusive}"
# Host side of the parked DiTs (16gb profile): "mmap" = bf16 safetensors under
# ACESTEP_DIT_RESIDENCY_DIR (default .cache/acestep/dit_residency) mapped into
# the page cache; "pinned" = locked RAM copies (~23 GB), swap ~0.3 s faster.
export ACESTEP_DIT_RESIDENCY_STORE="${ACESTEP_DIT_RESIDENCY_STORE:-mmap}"
export ACESTEP_QUANTIZATION="${ACESTEP_QUANTIZATION:-none}"
export ACESTEP_COMPILE_MODEL="${ACESTEP_COMPILE_MODEL:-0}"
export ACESTEP_FUSED_RMSNORM="${ACESTEP_FUSED_RMSNORM:-exact}"

export ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-1}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-0.6B}"
export ACESTEP_LM_BACKEND="${ACESTEP_LM_BACKEND:-pt}"

export ACESTEP_DEVICE="${ACESTEP_DEVICE:-auto}"
export ACESTEP_OFFLOAD_TO_CPU="${ACESTEP_OFFLOAD_TO_CPU:-1}"
export ACESTEP_RESIDENT_COMPONENTS="${ACESTEP_RESIDENT_COMPONENTS-vae}"
export ACESTEP_LM_OFFLOAD_TO_CPU="${ACESTEP_LM_OFFLOAD_TO_CPU:-0}"
export ACESTEP_OFFLOAD_DIT_TO_CPU="${ACESTEP_OFFLOAD_DIT_TO_CPU:-0}"
export ACESTEP_VAE_DECODE_CHUNK_SIZE="${ACESTEP_VAE_DECODE_CHUNK_SIZE:-512}"
# export ACESTEP_MP3_COMPRESSION_LEVEL=7   # optional LAME speed preset (-1.7 s / 300 s song)
export ACESTEP_NO_INIT="${ACESTEP_NO_INIT:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Concurrency (see 運用メモ_同時実行数とキュー.md): ACESTEP_QUEUE_WORKERS jobs run at
# once, ACESTEP_API_WORKERS must match. Jobs on the DiT that is on the GPU run in
# parallel; a request for another DiT waits for them, swaps, then runs. LM calls
# are serialised internally. With more than one worker the text encoder must
# stay on the GPU too (set below), so budget VRAM for the extra activations.
export ACESTEP_QUEUE_WORKERS="${ACESTEP_QUEUE_WORKERS:-1}"
export ACESTEP_API_WORKERS="${ACESTEP_API_WORKERS:-$ACESTEP_QUEUE_WORKERS}"
if [[ "$ACESTEP_QUEUE_WORKERS" -gt 1 && "${ACESTEP_RESIDENT_COMPONENTS}" == "vae" ]]; then
	export ACESTEP_RESIDENT_COMPONENTS="vae,text_encoder"
fi
export ACESTEP_QUEUE_MAXSIZE="${ACESTEP_QUEUE_MAXSIZE:-200}"
HOST="${ACESTEP_API_HOST:-0.0.0.0}"
PORT="${ACESTEP_API_PORT:-8001}"
LOG_LEVEL="${ACESTEP_API_LOG_LEVEL:-info}"

exec python -m uvicorn acestep.api_server:app \
	--host "$HOST" \
	--port "$PORT" \
	--workers 1 \
	--log-level "$LOG_LEVEL"
