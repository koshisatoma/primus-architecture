#!/bin/bash
set -e
echo "=== PRIMUS ASTRA & MJX TRAINING ==="
echo "Wallet check: Ensure you have ₹200-300+ before running!"

# Ensure correct path and environment
cd /home/ryu/primus3
source cx/bin/activate
export PYTHONPATH=/home/ryu/primus3:$PYTHONPATH

# 1. Collect ASTRA Episodes
echo "[1/3] Collecting ASTRA episodes (saves to data/astra/)..."
python -m cloud.collect_episodes --eps 50 --T 200 --out data/astra/episodes

# 2. Train ASTRA World Model
echo "[2/3] Training ASTRA World Model..."
python -m cloud.train_astra --epochs 20 --align_epochs 2 --out data/astra/astra_grounding.pt

# 3. Train MJX Stand Policy
echo "[3/3] Training MJX Stand Policy..."
python -m cloud.train_reflex_ppo --task stand --envs 1024 --steps 200000 --T 128 --out data/policies/ckpt
python -m cloud.export_policy --ckpt data/policies/ckpt_stand.npz --out data/policies/stand_policy

echo "========================================="
echo "TRAINING COMPLETE!"
echo "Results saved to:"
echo "  - data/astra/episodes.npz"
echo "  - data/astra/astra_grounding.pt"
echo "  - data/policies/ckpt_stand.npz"
echo ""
echo "Push to GitHub and terminate:"
echo "  git add data/"
echo "  git commit -m 'A100 training: ASTRA + Stand policy'"
echo "  git push"
echo "========================================="
