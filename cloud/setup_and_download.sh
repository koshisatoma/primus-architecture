#!/bin/bash
set -e
echo "=== PRIMUS SETUP & MODEL DOWNLOAD ==="

# 1. System Dependencies
echo "[1/4] Installing system dependencies..."
sudo apt-get update -qq && sudo apt-get install -y -qq build-essential gcc g++ libgl1-mesa-glx libxcb1 libx11-6 libsm6 libxext6 > /dev/null 2>&1
echo "System dependencies installed."

# 2. Python Environment & Core Packages
echo "[2/4] Setting up Python environment..."
if [ ! -d "cx" ]; then
    python3 -m venv cx
fi
source cx/bin/activate
pip install --upgrade pip -q

# Install verified MJX + JAX + PyTorch stack
pip install "mujoco==3.2.2" mujoco-mjx "jax[cuda12]" numpy torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install faster-whisper ultralytics opencv-python-headless fastapi uvicorn gradio rich huggingface_hub transformers -q

# Verify MJX works
python -c "import jax; import mujoco.mjx; print('MJX Verified on', jax.devices()[0])"
echo "Python packages installed."

# 3. Download Models (~25GB) DIRECTLY INTO THIS PROJECT FOLDER
echo "[3/4] Downloading models to ./models (10-15 mins)..."
mkdir -p models
python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

models = {
    'whisper': 'Systran/faster-whisper-large-v3',
    'depth': 'depth-anything/Depth-Anything-V2-Large-hf',
    'dinov2': 'facebook/dinov2-large',
    'qwen': 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'
}

for name, repo in models.items():
    print(f'  Downloading {name} to ./models/{name}...')
    snapshot_download(repo_id=repo, local_dir=f'models/{name}')
print('All models downloaded to project folder.')
"

# 4. Create Organized Data Folders
echo "[4/4] Creating organized data folders..."
mkdir -p data/{astra,detection/{known,unknown},recordings,policies}

# Initialize JSON files
echo '{}' > data/detection/known/people.json
echo '{}' > data/detection/unknown/faces.json
echo '{}' > data/objects.json
echo '{}' > data/metrics.json

echo "========================================="
echo "SETUP & DOWNLOAD COMPLETE!"
echo "Models are safely stored in: /home/ryu/primus3/models/"
echo "Data folders created:"
echo "  - data/astra/ (ASTRA episodes & models)"
echo "  - data/detection/known/ (recognized people)"
echo "  - data/detection/unknown/ (temporary faces)"
echo "  - data/recordings/ (task teaching videos + JSON)"
echo "  - data/policies/ (trained checkpoints)"
echo ""
echo "Next: Add ₹200-300 to wallet, then run:"
echo "  ./cloud/train_all.sh"
echo "========================================="
