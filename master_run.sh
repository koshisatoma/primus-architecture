#!/bin/bash
set -e
echo "=== PRIMUS CLOUD MASTER RUN (A100) ==="

# 1. Pull ASTRA weights and models from Git LFS
echo "[1/3] Pulling LFS assets (ASTRA weights + Models)..."
git lfs pull

# 2. Setup Environment (if not already done)
if [ ! -d "models/whisper" ]; then
    echo "Models missing. Running setup..."
    ./cloud/setup_and_download.sh
else
    echo "Models already present."
fi

# 3. MJX Training (The heavy lifting for the A100)
echo "[2/3] Training MJX Stand Policy (Cloud A100)..."
source cx/bin/activate
export PYTHONPATH=/home/ryu/primus3:$PYTHONPATH

python -m cloud.train_reflex_ppo --task stand --envs 1024 --steps 200000 --T 128 --out data/policies/ckpt
python -m cloud.export_policy --ckpt data/policies/ckpt_stand.npz --out data/policies/stand_policy

# 4. Save to LFS
echo "[3/3] Saving to Git LFS..."
git add data/policies/
git commit -m "cloud run: MJX stand policy trained"
git push

echo "CLOUD MASTER RUN COMPLETE!"
