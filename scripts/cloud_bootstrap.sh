#!/usr/bin/env bash
set -euo pipefail

workdir="${PRIMUS_WORKDIR:-$HOME}"
repo_dir="${PRIMUS_REPO_DIR:-$workdir/primus-architecture}"
venv_dir="${PRIMUS_VENV:-$workdir/cx}"
python_bin="${PYTHON:-python3}"

mkdir -p "$workdir"
"$python_bin" -m venv "$venv_dir"
# shellcheck disable=SC1090
source "$venv_dir/bin/activate"

python -m pip install --upgrade pip
python -m pip install \
    'jax[cuda12]' \
    mujoco \
    mujoco-mjx \
    numpy \
    torch \
    torchvision \
    fastapi \
    uvicorn

if [[ -d "$repo_dir/.git" ]]; then
    git -C "$repo_dir" pull --ff-only
else
    git clone https://github.com/koshisatoma/primus-architecture.git "$repo_dir"
fi

cd "$repo_dir"
nvidia-smi
python -c "import jax; print('JAX devices:', jax.devices())"
echo "BOOTSTRAP DONE $(date)"
