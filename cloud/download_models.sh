#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="${1:-${MODELS:-$HOME/models}}"
venv_dir="${PRIMUS_VENV:-$HOME/cx}"
mkdir -p "$models_dir"

if [[ -f "$venv_dir/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
fi

if ! command -v hf >/dev/null 2>&1; then
    python -m pip install --upgrade huggingface_hub
fi

echo "Downloading model weights to $models_dir (internet and disk only; no GPU required)..."
hf download Qwen/Qwen2.5-VL-7B-Instruct-AWQ --local-dir "$models_dir/qwen-vl-7b"
hf download kyutai/moshi --local-dir "$models_dir/moshi"
hf download Systran/faster-whisper-large-v3 --local-dir "$models_dir/whisper-large-v3"
hf download depth-anything/Depth-Anything-V2-Large-hf --local-dir "$models_dir/depth-anything"
hf download facebook/dinov2-large --local-dir "$models_dir/dinov2"

echo "ALL MODELS ON DISK: $models_dir"
du -sh "$models_dir"/*
