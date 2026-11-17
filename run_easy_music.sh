#!/usr/bin/env bash
# Start Easy Music (the bundled web app, easy-music/) against the local API server.
#   ./run_easy_music.sh                 # http://localhost:8888, talks to http://127.0.0.1:8001
#   ./run_easy_music.sh 8890            # listen on another port (same as --port 8890)
#   ACE_STEP_API_URL=http://host:8001 ./run_easy_music.sh
# Environment: easy-music/.venv (created by ./setup.sh). Lyrics use an OpenAI-compatible LLM
# (OPENAI_BASE_URL, Ollama by default) with a CPU GGUF fallback; background images need an
# optional image server (IMAGE_GENERATION_URL). Settings: easy-music/.env (see its README).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/easy-music"
if [[ ! -f .venv/bin/activate ]]; then
	echo "easy-music/.venv is missing: run ./setup.sh (or ./setup.sh --no-models) first." >&2
	exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export ACE_STEP_API_URL="${ACE_STEP_API_URL:-http://127.0.0.1:${ACESTEP_API_PORT:-8001}}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then  # a bare number = listen port
	set -- --port "$1" "${@:2}"
fi
exec python main.py "$@"
