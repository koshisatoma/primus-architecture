#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-}"
if [[ -z "$python_bin" && -x "$root/.venv/bin/python" ]]; then
    python_bin="$root/.venv/bin/python"
elif [[ -z "$python_bin" ]]; then
    python_bin=python3
fi

cd "$root"
"$python_bin" -m unittest discover -s tests -p 'test_human_behavior.py'
