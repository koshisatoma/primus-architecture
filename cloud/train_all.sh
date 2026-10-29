#!/bin/bash
set -e
echo "=== LOCAL ASTRA TRAINING (8GB GPU SAFE) ==="

# Ensure correct path and environment
cd /home/ryu/primus3
source .venv/bin/activate 2>/dev/null || source cx/bin/activate
export PYTHONPATH=/home/ryu/primus3:$PYTHONPATH

# CRITICAL: Prevent JAX from hoarding all 8GB of your local VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80

# 1. Collect ASTRA Episodes (CPU/RAM heavy, low VRAM)
echo "[1/2] Collecting ASTRA episodes (saves to data/astra/)..."
python -m cloud.collect_episodes --eps 10 --T 200 --out data/astra/episodes

# 2. Train ASTRA World Model (Uses ~4-6 GB VRAM)
echo "[2/2] Training ASTRA World Model..."
python -m cloud.train_astra --epochs 10 --align_epochs 2 --out data/astra/astra_grounding.pt
