# PRIMUS Setup Guide (From Scratch)

This document provides **exact, copy-pasteable commands** to set up PRIMUS on both local machines and cloud GPU nodes. Follow these steps precisely to replicate the development environment.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Setup (Ubuntu Laptop)](#2-local-setup-ubuntu-laptop)
3. [Cloud GPU Setup (A100/A6000)](#3-cloud-gpu-setup-a100a6000)
4. [VSCode Remote-SSH Configuration](#4-vscode-remote-ssh-configuration)
5. [Model Downloads](#5-model-downloads)
6. [Verification & Testing](#6-verification--testing)
7. [Running the System](#7-running-the-system)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### 1.1 Local Machine Requirements
- **OS**: Ubuntu 22.04 or 24.04
- **CPU**: 8+ cores (Intel i7/Ryzen 7 or better)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or higher (for local smoke tests, teleop, UI)
- **Storage**: 50GB free space
- **Network**: Stable internet

### 1.2 Cloud GPU Requirements
- **Provider**: E2E Networks, JarvisLabs, or similar with A100 (80GB) or RTX A6000 (48GB)
- **OS Image**: Ubuntu 22.04 with PyTorch 2.x + CUDA 12 template
- **Storage**: 100GB+ persistent SSD
- **Budget**: ~₹1,000 for full training run (~10 hours on A100)

### 1.3 Software Dependencies
- **Python**: 3.10+
- **Git**: 2.30+
- **CUDA**: 12.1+
- **CMake**: 3.20+
- **Rust**: 1.70+

---

## 2. Local Setup (Ubuntu Laptop)

### 2.1 System Dependencies
```bash
sudo apt update
sudo apt install -y build-essential gcc g++ libgl1-mesa-glx libxcb1 libx11-6 libsm6 libxext6 libxrender1 libxfixes3 libxrandr2 python3.10 python3.10-venv python3.10-dev git curl wget
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2.2 Clone Repository
```bash
cd ~
git clone https://github.com/koshisatoma/primus-architecture.git primus3
cd primus3
```

### 2.3 Create Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2.4 Install Python Packages
```bash
pip install "mujoco>=3.0" "jax[cuda12]" numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm faster-whisper ultralytics opencv-python-headless transformers fastapi uvicorn gradio rich
```

### 2.5 Compile C++ Spine
```bash
MUJOCO_DIR=$(python -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")
g++ core_cpp/spine.cpp -O2 -I$MUJOCO_DIR/include -L$MUJOCO_DIR/lib -lmujoco -o spine_bin -Wl,-rpath,$MUJOCO_DIR/lib
```

### 2.6 Build Rust Safety Watchdog
```bash
cd safety_rs && cargo build --release && cd ..
```

---

## 3. Cloud GPU Setup (A100/A6000)

### 3.1 One-Click Setup Script
SSH into your cloud node and run the master script:
```bash
ssh root@<YOUR_CLOUD_IP>
mkdir /primus3 && cd /primus3
git clone https://github.com/koshisatoma/primus-architecture.git .
chmod +x cloud/master_run.sh
./cloud/master_run.sh
```
*This script installs system deps, Python packages, downloads ~25GB of models, collects ASTRA episodes, trains the world model, and trains one primitive.*

### 3.2 Manual Setup (If Needed)
```bash
apt-get update && apt-get install -y build-essential gcc g++ libgl1-mesa-glx libxcb1 libx11-6
python3 -m venv cx && source cx/bin/activate
pip install --upgrade pip
pip install --upgrade "mujoco>=3.0" "jax[cuda12]" numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm faster-whisper ultralytics fastapi uvicorn opencv-python-headless gradio rich
```

---

## 4. VSCode Remote-SSH Configuration

### 4.1 Install Extension
On your local machine, install the **"Remote - SSH"** extension by Microsoft in VSCode.

### 4.2 Configure Host
1. Press `Ctrl+Shift+P` → **Remote-SSH: Connect to Host...** → **Add New SSH Host...**
2. Enter: `ssh root@<YOUR_CLOUD_IP>`
3. Select your `~/.ssh/config` file to update.

### 4.3 Connect
1. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host...** → select your IP.
2. Select **Linux**. Enter password or use SSH key.
3. Bottom-left corner will show `SSH: <YOUR_IP>` in green.
4. Press `Ctrl+O` and open the `/primus3` folder.
5. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `/primus3/cx/bin/python`.

---

## 5. Model Downloads

Models are downloaded automatically by `master_run.sh`. If doing manually:
```bash
mkdir -p /root/models
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Systran/faster-whisper-large-v3', local_dir='/root/models/whisper-large-v3')
snapshot_download(repo_id='depth-anything/Depth-Anything-V2-Large-hf', local_dir='/root/models/depth-anything')
snapshot_download(repo_id='facebook/dinov2-large', local_dir='/root/models/dinov2')
snapshot_download(repo_id='Qwen/Qwen2.5-VL-7B-Instruct-AWQ', local_dir='/root/models/qwen-vl-7b')
print('ALL MODELS DOWNLOADED')
"
```

---

## 6. Verification & Testing

### 6.1 Local Smoke Test
```bash
cd ~/primus3 && source .venv/bin/activate
python -c "from mujoco import mjx; print('MJX OK')"
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
python tests/test_human_likeness_battery.py
```

### 6.2 Cloud Training Test
```bash
cd /primus3 && source cx/bin/activate
python -m cloud.train_reflex_ppo --task stand --envs 512 --steps 10000 --T 128 --out data/ckpt
python -m cloud.collect_episodes --eps 5 --T 50 --out data/episodes
python -m cloud.train_astra --epochs 2 --align_epochs 1
ls -lh data/ckpt_stand.npz data/astra_grounding.pt
```

---

## 7. Running the System

### 7.1 Local UI
```bash
cd ~/primus3 && source .venv/bin/activate
python3 app.py
# Open http://127.0.0.1:7860
```

### 7.2 Cloud Full Training
```bash
for TASK in stand walk walk_to sit reach grasp lift place; do
    python -m cloud.train_reflex_ppo --task $TASK --envs 2048 --steps 500000 --T 256 --out data/ckpt
    python -m cloud.export_policy --ckpt data/ckpt_${TASK}.npz --out data/${TASK}_policy
done
python -m cloud.collect_episodes --eps 200 --T 500 --out data/episodes
python -m cloud.train_astra --epochs 50 --align_epochs 20
git add data/ && git commit -m "Trained all primitives + ASTRA grounding" && git push
```

### 7.3 Teleop Teaching (Local)
```bash
python -m cloud.teleop --task reach --policy data/reach_policy_weights.npz --out data/demos
# Controls: 1/2/3/4 = joint group, Arrows = move, SPACE = takeover, S = save
```

### 7.4 Nero Gesture Pick-and-Place Teaching
```bash
python -m teleop.gesture_teach --task stack_cubes
python -m teleop.gesture_teach --task move_cup
python -m teleop.gesture_teach --task glass_to_plate
```

The fullscreen tabletop simulation includes both cubes and the open box, the
cup and its marked destination, and the glass and plate. Drag the preview to
orbit around the scene and use the mouse wheel to zoom. Hand gestures move and
lower the arm, pinch to pick up an object, open the hand to place it, and make
a fist to save the demonstration. Use `--camera N` to select a different
webcam; use `--overwrite` only when intentionally replacing an existing
recording. If the camera cannot be opened, close other camera applications,
check system camera permissions, or try another camera index. Without a
webcam, the tool automatically falls back to keyboard teaching: use arrow
keys to move, `W`/`S` or Page Up/Down to raise/lower, `P` to pick, `O` to
place, **Shift + arrows** to nudge the held object, and Enter to save. Pass
`--keyboard` to select this mode directly.

Replay saved tasks in the 3D window:
```bash
python -m teleop.executor --task move_cup
python -m teleop.executor --task glass_to_plate
```

Click **Reset / Replay** (or press `R`) to run it again; Space pauses or
resumes, and `+`/`-` adjust playback speed.

---

## 8. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'mjx'` | MuJoCo < 3.0 | `pip install --upgrade "mujoco>=3.0"` |
| `ImportError: libxcb.so.1` | Missing OpenCV deps | `apt-get install -y libxcb1 libx11-6` + `pip install opencv-python-headless` |
| `Failed to find C compiler` | vLLM Triton needs gcc | `apt-get install -y build-essential gcc g++` |
| `vLLM Engine core initialization failed` | V1 multiprocessing bug | `export VLLM_ENABLE_V1_MULTIPROCESSING=0` and add `--enforce-eager` |
| `HuggingFace HTTP 401` | Rate limits / deprecated CLI | Use Python `snapshot_download` script |

---

**Setup complete!** Proceed to [README.md](README.md) for architecture details.