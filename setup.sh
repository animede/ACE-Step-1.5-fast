#!/usr/bin/env bash
# ACE-Step-1.5-fast setup: uv -> Python environment -> checkpoints -> GPU check.
# Linux x86_64 + NVIDIA (CUDA 12.8 wheels). Re-runnable; existing models are skipped.
#
#   ./setup.sh            # 16gb profile set: xl-sft, xl-turbo, 2B turbo, LM 0.6B (+ vae, text encoder)  ~45 GB
#   ./setup.sh --large    # also LM 4B (ACESTEP_GPU_PROFILE=large)                                        ~53 GB
#   ./setup.sh --no-models   # environment only
#   ./setup.sh --no-client   # skip the client-pro web app environment (client-pro/.venv)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

WANT_MODELS=1; LARGE=0; WANT_CLIENT=1
for arg in "$@"; do
	case "$arg" in
		--no-models) WANT_MODELS=0 ;;
		--no-client) WANT_CLIENT=0 ;;
		--large) LARGE=1 ;;
		-h|--help) sed -n '2,8p' "$0"; exit 0 ;;
		*) echo "unknown option: $arg" >&2; exit 1 ;;
	esac
done

echo "[1/5] uv"
if ! command -v uv >/dev/null 2>&1; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version

echo "[2/5] Python environment (.venv, from uv.lock)"
uv sync --frozen

echo "[3/5] GPU / libraries"
uv run --no-sync python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda available: {ok}")
if ok:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  {p.total_memory / 2**30:.1f} GB  sm_{p.major}{p.minor}")
try:
    import flash_attn
    print(f"flash_attn {flash_attn.__version__} (used by the DiT attention)")
except Exception as exc:  # the server falls back to SDPA, ~10-25% slower
    print(f"flash_attn not importable ({exc}); the server will use SDPA")
PY

if [[ "$WANT_MODELS" == "1" ]]; then
	echo "[4/5] checkpoints (skips what already exists)"
	uv run --no-sync acestep-download --model acestep-v15-xl-sft      # 19 GB (also fetches the main set: vae, text encoder, 2B turbo, LM 1.7B)
	uv run --no-sync acestep-download --model acestep-v15-xl-turbo    # 19 GB
	uv run --no-sync acestep-download --model acestep-5Hz-lm-0.6B     # 1.3 GB
	if [[ "$LARGE" == "1" ]]; then
		uv run --no-sync acestep-download --model acestep-5Hz-lm-4B   # 7.9 GB
	fi
else
	echo "[4/5] checkpoints: skipped (--no-models)"
fi

if [[ "$WANT_CLIENT" == "1" ]]; then
	echo "[5/5] client-pro web app environment (client-pro/.venv)"
	[[ -d client-pro/.venv ]] || uv venv client-pro/.venv --python 3.11 -q
	uv pip install --python client-pro/.venv/bin/python -q -r <(grep -v "llama-cpp-python" client-pro/requirements.txt)
	# CPU-only prebuilt wheel for the GGUF lyrics fallback (no compiler needed); optional
	uv pip install --python client-pro/.venv/bin/python -q "llama-cpp-python>=0.3.0" \
		--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
		|| echo "  llama-cpp-python not installed: the client will use the external LLM only"
else
	echo "[5/5] client-pro environment: skipped (--no-client)"
fi

echo
echo "Done. Start the server with:  ./run_api_server_fast.sh   (then open http://localhost:8001/ui/)"
echo "      and the web app with:   ./run_client_pro.sh        (http://localhost:8888)"
