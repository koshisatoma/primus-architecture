#!/bin/bash
set -e
echo "=== PRIMUS A100 MASTER RUN ==="

# 1. System Dependencies (Fixes the AIC Cloud bugs)
echo "[1/6] Installing system deps..."
apt-get update -qq && apt-get install -y -qq build-essential gcc g++ libgl1-mesa-glx libxcb1 libx11-6 > /dev/null

# 2. Python Environment
echo "[2/6] Setting up Python env..."
if [ ! -d "cx" ]; then
    python3 -m venv cx
fi
source cx/bin/activate
pip install --upgrade pip -q
pip install --upgrade "mujoco>=3.0" "jax[cuda12]" numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
pip install vllm faster-whisper ultralytics fastapi uvicorn opencv-python-headless gradio rich -q

# 3. Download Models (Disk only, ~25GB)
echo "[3/6] Downloading models..."
mkdir -p /root/models
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Systran/faster-whisper-large-v3', local_dir='/root/models/whisper-large-v3')
snapshot_download(repo_id='depth-anything/Depth-Anything-V2-Large-hf', local_dir='/root/models/depth-anything')
snapshot_download(repo_id='facebook/dinov2-large', local_dir='/root/models/dinov2')
snapshot_download(repo_id='Qwen/Qwen2.5-VL-7B-Instruct-AWQ', local_dir='/root/models/qwen-vl-7b')
print('Models downloaded.')
"

# 4. Collect Episodes for ASTRA
echo "[4/6] Collecting episodes..."
python -m cloud.collect_episodes --eps 50 --T 200 --out data/episodes

# 5. Train ASTRA (Algorithms 1 & 2)
echo "[5/6] Training ASTRA World Model..."
python -m cloud.train_astra --epochs 20 --align_epochs 5

# 6. Train ONE Primitive (Stand) to save time
echo "[6/6] Training Stand Policy..."
python -m cloud.train_reflex_ppo --task stand --envs 1024 --steps 200000 --T 128 --out data/ckpt
python -m cloud.export_policy --ckpt data/ckpt_stand.npz --out data/stand_policy

echo "=== RUN COMPLETE ==="
echo "Next: git add data/ && git commit -m 'A100 run' && git push"