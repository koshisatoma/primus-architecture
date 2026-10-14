#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-}"

if [[ -z "$python_bin" && -x "$root/.venv/bin/python" ]]; then
    python_bin="$root/.venv/bin/python"
elif [[ -z "$python_bin" ]]; then
    python_bin=python
fi

cd "$root"
"$python_bin" -m unittest discover -s "$root/tests"

"$python_bin" -m py_compile \
    "$root/cloud/__init__.py" \
    "$root/cloud/g1_env.py" \
    "$root/cloud/train_reflex_ppo.py" \
    "$root/cloud/collect_episodes.py" \
    "$root/cloud/train_astra.py" \
    "$root/cloud/export_policy.py" \
    "$root/cloud/svc_percept.py" \
    "$root/primus/cloud_client.py"

"$python_bin" -c \
    "import jax; import mujoco; from mujoco import mjx; import torch"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/primus-cloud-smoke.XXXXXX")"
trap 'rm -rf -- "$workdir"' EXIT

"$python_bin" -m cloud.train_reflex_ppo \
    --task stand --envs 2 --steps 16 --T 8 --out "$workdir/ckpt"
"$python_bin" -m cloud.collect_episodes \
    --eps 2 --T 8 --out "$workdir/episodes"
"$python_bin" -m cloud.train_astra \
    --data "$workdir/episodes.npz" --epochs 1 --align_epochs 1 \
    --out "$workdir/astra_grounding.pt"
"$python_bin" -m cloud.export_policy \
    --ckpt "$workdir/ckpt_stand.npz" --out "$workdir/stand_policy"

echo "SMOKE TEST PASSED - local CPU/GPU training stack verified"
