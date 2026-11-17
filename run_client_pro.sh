#!/usr/bin/env bash
# Start ACE-Step-client-pro (the reference web app, client-pro/) against the local API server.
#   ./run_client_pro.sh                 # http://localhost:8888, talks to http://127.0.0.1:8001
#   ./run_client_pro.sh 8890            # listen on another port (same as --port 8890)
#   ACE_STEP_API_URL=http://host:8001 ./run_client_pro.sh
# The client has its own environment (client-pro/.venv, created by ./setup.sh). Lyrics use an
# OpenAI-compatible LLM (OPENAI_BASE_URL, Ollama by default) with a CPU GGUF fallback.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/client-pro"
if [[ ! -f .venv/bin/activate ]]; then
	echo "client-pro/.venv is missing: run ./setup.sh (or ./setup.sh --no-models) first." >&2
	exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export ACE_STEP_API_URL="${ACE_STEP_API_URL:-http://127.0.0.1:${ACESTEP_API_PORT:-8001}}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then  # a bare number = listen port
	set -- --port "$1" "${@:2}"
fi
exec python main.py "$@"
