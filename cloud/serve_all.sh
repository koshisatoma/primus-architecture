#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="${MODELS:-$HOME/models}"
venv_dir="${PRIMUS_VENV:-$HOME/cx}"

if [[ ! -d "$models_dir/qwen-vl-7b" ]]; then
    echo "Missing model directory: $models_dir/qwen-vl-7b (run cloud/download_models.sh first)" >&2
    exit 1
fi

cd "$root"
if [[ -f "$venv_dir/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
fi

python -m vllm.entrypoints.openai.api_server \
    --model "$models_dir/qwen-vl-7b" \
    --served-model-name qwen-vl \
    --host "${PRIMUS_BIND_HOST:-0.0.0.0}" \
    --port 8000 \
    --gpu-memory-utilization "${PRIMUS_BRAIN_GPU_MEMORY_UTILIZATION:-0.45}" \
    --max-model-len "${PRIMUS_BRAIN_MAX_MODEL_LEN:-4096}" &
brain_pid=$!

python -m moshi.server --port 8001 &
voice_pid=$!

python -m cloud.svc_percept &
percept_pid=$!

cleanup() {
    kill "$brain_pid" "$voice_pid" "$percept_pid" 2>/dev/null || true
    wait "$brain_pid" "$voice_pid" "$percept_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Services starting: 8000 brain / 8001 voice / 8002 percept"
set +e
wait -n "$brain_pid" "$voice_pid" "$percept_pid"
status=$?
exit "$status"
