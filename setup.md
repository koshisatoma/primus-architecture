# Setup and model guide

This guide covers the local application, optional model downloads, GPU cloud
services/training, and ROS 2. The project has no root `requirements.txt` or
`pyproject.toml`, so clean installs are not currently pinned to one fully
reproducible dependency set. Keep the Python version and accelerator packages
compatible with your operating system and hardware.

For the project and architecture overview, start at [README.md](./README.md).
For cloud-only commands, see [cloud/README.md](./cloud/README.md).

## 1. Clone and create a local Python environment

The checked-in workspace has been developed with Python 3.10 on Linux. A
virtual environment keeps application packages isolated:

```bash
git clone https://github.com/koshisatoma/primus-architecture.git
cd primus-architecture
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the application packages:

```bash
python -m pip install \
  gradio mujoco numpy opencv-python ultralytics \
  faster-whisper sounddevice piper-tts requests
```

On Debian/Ubuntu, SoundDevice may need PortAudio system libraries:

```bash
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev
```

The exact PyTorch wheel depends on CPU/CUDA and driver versions. Ultralytics
normally installs a compatible PyTorch package, but for NVIDIA deployments
follow the PyTorch installation selector and confirm the installed CUDA
runtime is compatible with the driver. No GPU is required for the basic UI or
CPU-compatible smoke test.

### Existing development environment

If the repository already has a working `.venv`, use it instead of creating a
second environment:

```bash
source .venv/bin/activate
python --version
```

Never commit the virtual environment. It is excluded from version control.

## 2. Local application models

### YOLO object/person detection

[`primus/detect.py`](./primus/detect.py) looks for model files in the repository
root in this priority order:

1. `yolo26x.engine` (YOLO26x TensorRT)
2. `yolo26x.pt` (YOLO26x PyTorch)
3. `yolo11x.engine` (YOLO11x TensorRT)
4. `yolo11x.pt` (YOLO11x PyTorch)
5. `yolo11n.engine` (YOLO11n TensorRT)
6. `yolo11n.pt` (YOLO11n PyTorch)

The first candidate that exists and loads successfully is used. A TensorRT
engine must have been built for a compatible GPU, TensorRT/CUDA stack, and
runtime. PyTorch `.pt` weights are generally the easier cross-machine choice.
To let Ultralytics download a default YOLO11 weight and verify that it loads,
run this from the repository root:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

Put the selected `.pt` or compatible `.engine` file at the repository root;
for any other supported candidate, use its official Ultralytics source or a
licensed local copy. Verify the license and terms for each model before
redistribution.

`yolo26n-seg.pt`, `yolo11n.onnx`, and `*.fp16.onnx` are not currently in the
local detector candidate list. The presence of an ONNX file does not make the
current detector use it. To inspect what loaded, call `primus.detect.model_kind()`
after first inference.

### Local voice recognition

The default local fallback is Faster Whisper `base`, initialized automatically
on the first transcription that needs the local model. It downloads model
weights into the Faster Whisper/Hugging Face cache; no separate project command
is required. The first use needs internet access and disk space. The current
local configuration uses CPU with int8 quantization.

### Local voice output (Piper)

The local output voice is expected at:

- `data/en_US-hfc_female-medium.onnx`
- `data/en_US-hfc_female-medium.onnx.json`

The app prints replies when the ONNX voice is absent or Piper is not installed.
The voice is not downloaded by `cloud/download_models.sh`. One way to fetch the
matching files from the Piper voices repository is:

```bash
python -m pip install --upgrade huggingface_hub
mkdir -p "$HOME/primus-piper-download"
hf download rhasspy/piper-voices \
  en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx \
  en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json \
  --local-dir "$HOME/primus-piper-download"
mkdir -p data
cp "$HOME/primus-piper-download/en/en_US/hfc_female/medium/"* data/
```

If the upstream repository changes its layout or requires access approval, use
the Piper voice catalog to locate the same voice and preserve the expected
filenames under `data/`. Review the voice model's license before redistribution.

The cognition integration also stores brain input/reply records (including
command and introduction turns) in `data/primus_memory.sqlite3`. This is
separate from the ignored `data/` directory already used for object and face data. Review
[`docs/HUMAN_BEHAVIOR.md`](./docs/HUMAN_BEHAVIOR.md) for memory controls and
prototype limitations.

### Local conversation model (Ollama)

When cloud chat is not configured or unavailable, the app uses a local Ollama
service with `qwen2.5:3b-instruct`:

```bash
# Install Ollama using the official instructions: https://ollama.com/download
ollama pull qwen2.5:3b-instruct
ollama serve
```

Keep Ollama running in another terminal. The app calls its local API at
`http://127.0.0.1:11434/api/chat`. Without Ollama, only built-in responses are
available for general chat. Ollama installation is separate from the Python
environment and is not performed by the project scripts.

## 3. Run the local app

Connect or select a camera and microphone, activate the environment, then:

```bash
source .venv/bin/activate
python app.py
```

Open <http://127.0.0.1:7860>. The UI renders the MuJoCo scene and telemetry.
The camera path prefers the C++ shared-memory camera if available and otherwise
tries OpenCV camera index 0. Voice runs asynchronously in a background thread.

The optional `cam_bin` executable reads camera index 0 and publishes shared
memory named `primus_cam`:

```bash
./cam_bin
```

`cam_bin` is a local build artifact and is not committed. The source is
[`core_cpp/cam.cpp`](./core_cpp/cam.cpp). This repository currently does not
include a root CMake project or a canonical rebuild command for the executable.

## 4. Optional cloud model downloads

The script downloads five model directories to `$HOME/models` by default. It
downloads weights only; it does not install all serving dependencies or launch
the services. Downloads need internet access, Hugging Face access where
required, and substantial disk space. No GPU is needed just to download.

```bash
source .venv/bin/activate
bash cloud/download_models.sh
```

Override the destination:

```bash
MODELS="$HOME/models" bash cloud/download_models.sh "$HOME/models"
```

The script uses Hugging Face CLI and writes:

| Directory | Hugging Face model | Purpose |
| --- | --- | --- |
| `qwen-vl-7b/` | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | vLLM chat service on port 8000 |
| `moshi/` | `kyutai/moshi` | Voice service on port 8001 |
| `whisper-large-v3/` | `Systran/faster-whisper-large-v3` | Cloud speech transcription |
| `depth-anything/` | `depth-anything/Depth-Anything-V2-Large-hf` | Depth estimation endpoint |
| `dinov2/` | `facebook/dinov2-large` | Person embedding/re-identification endpoint |

Some model repositories may require accepting terms or authenticating with
Hugging Face. Authenticate using the official `hf auth login` workflow if
required; do not put access tokens in source files or commit them.

`cloud/svc_percept.py` also needs a YOLO model for person detection. It defaults
to `yolo11x.pt` or uses `PRIMUS_YOLO_MODEL` if set. That weight is **not**
downloaded by `cloud/download_models.sh`; acquire it separately and make it
available from the service working directory or configure the model path.

## 5. Optional cloud serving host

### Training base environment

On a compatible Linux/NVIDIA GPU host, the bootstrap script creates the
`$HOME/cx` environment and installs the base training/service packages,
then clones or fast-forwards this repository:

```bash
bash scripts/cloud_bootstrap.sh
```

Override paths when needed:

```bash
PRIMUS_WORKDIR="$HOME" \
PRIMUS_REPO_DIR="$HOME/primus-architecture" \
PRIMUS_VENV="$HOME/cx" \
bash scripts/cloud_bootstrap.sh
```

The script installs `jax[cuda12]`, MuJoCo, MuJoCo-MJX, NumPy, PyTorch,
TorchVision, FastAPI, and Uvicorn, and then checks `nvidia-smi` and JAX devices.
Use a host whose installed driver supports the selected CUDA/JAX wheels.

### Serving dependencies and launch

The bootstrap script is primarily a training base, not a complete serving
image. For cloud serving, install compatible versions of vLLM, Moshi,
Transformers, Faster Whisper, OpenCV, Ultralytics, and Hugging Face Hub into
the same environment. GPU package versions vary by driver and model; use the
upstream installation instructions for those projects.

Download models as in the previous section, then launch:

```bash
export MODELS="$HOME/models"
export PRIMUS_VENV="$HOME/cx"
bash cloud/serve_all.sh
```

The service processes bind to `0.0.0.0` by default. Restrict firewall/security
group access to trusted clients or use an authenticated private gateway. These
services do not implement authentication.

On the client machine, set the cloud base address (without a port; the client
selects the service port):

```bash
export PRIMUS_POD="http://<private-host-or-tunnel-address>"
python app.py
```

The local client uses chat port 8000 and transcription port 8002. If a request
fails, it logs a warning and falls back to local chat/Whisper where available.

### Cloud service endpoints

- `POST /v1/chat/completions` on port 8000: OpenAI-compatible vLLM route.
- `POST /transcribe` on port 8002: little-endian signed 16-bit PCM audio,
  mono 16 kHz from the current client.
- `POST /depth` on port 8002: encoded image bytes; returns little-endian
  `uint16` depth values. Width, height, and dtype are response headers.
- `POST /reid` on port 8002: encoded image bytes; returns detected person boxes
  and DINOv2 embeddings.

The client currently calls chat and transcription; depth/re-identification are
service endpoints, not a complete integrated cloud perception pipeline.

## 6. Optional PPO, ASTRA, and policy export

Install/use the training dependencies in the chosen environment. For a quick
sanity check, run:

```bash
bash cloud/smoke_test.sh
```

The smoke script runs unit tests, syntax/import checks, and small PPO, episode,
ASTRA, and policy-export jobs. It supports CPU execution for the smoke-sized
configuration; large runs benefit from a compatible GPU.

Example PPO training:

```bash
python -m cloud.train_reflex_ppo \
  --task stand --envs 2048 --steps 800000 --out data/ckpt
```

PPO task choices currently include `stand`, `walk`, `walk_to`, `sit`, `reach`,
`grasp`, `lift`, and `place`. The CLI does not currently accept a grounded target
position or object body. Consequently, do not treat all task choices as
task-grounded training recipes: the reach/grasp/lift/place reward paths need
explicit target/object configuration and validation before meaningful use.

Collect episodes and train ASTRA:

```bash
python -m cloud.collect_episodes --eps 200 --T 500 --out data/episodes
python -m cloud.train_astra \
  --data data/episodes.npz --epochs 50 --align_epochs 20 \
  --out data/astra_grounding.pt
```

The episode collector currently records stand-policy simulation experience.
ASTRA trains a source encoder, transition/reward prediction heads, an absolute
state correction head, and an observation target encoder. It is not yet evidence
of a complete real-time latent MPC system.

Export a trained policy:

```bash
python -m cloud.export_policy \
  --ckpt data/ckpt_stand.npz --out data/stand_policy
```

The exporter writes an NPZ weights file and a C++-friendly directory of flat
layer binaries and a policy specification. These artifacts are not bundled in
Git; generate and verify them on the target host.

`cloud/wm_planner.py` is an experimental standalone planner and is not currently
connected to the 500 Hz reflex spine or the application. Its present checkpoint
loader expects a model object with `plat`/`rpred` methods, whereas
`cloud/train_astra.py` saves a dictionary containing a `model_state_dict`.
An explicit adapter/checkpoint-loading implementation is needed before using
those artifacts together. The documented 10 Hz/50 ms budget is a design target,
not a validated real-time guarantee.

## 7. Optional ROS 2 Humble

Install ROS 2 Humble and `colcon` using the official ROS instructions. In a new
terminal:

```bash
cd primus_ros2
source /opt/ros/humble/setup.bash
colcon build --packages-select primus_actions
source install/setup.bash
export PRIMUS3_ROOT="$(cd .. && pwd)"
```

Start each node in its own terminal with ROS setup sourced:

```bash
ros2 run primus_actions perception_bridge
ros2 run primus_actions task_executor
ros2 run primus_actions brain_bridge
```

Send a text command to test routing:

```bash
ros2 topic pub --once /primus/voice_text std_msgs/msg/String \
  "{data: 'find the cup'}"
```

The ROS package is a bridge/skeleton. Perception and command routing are
available, but there is no implemented navigation, locomotion, grasp, or place
backend. `task_executor` reports that the motion backend is unavailable.

## 8. Tests and troubleshooting

Run the unit tests:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

Run the complete cloud smoke script:

```bash
bash cloud/smoke_test.sh
```

Common issues:

- **No camera image:** check camera permissions and device index 0. For shared
  memory, start `cam_bin` before the app.
- **No speech output:** confirm the Piper voice files are in `data/`, or expect
  the response to print to the terminal.
- **First local transcription is slow:** Faster Whisper downloads and initializes
  the `base` model on first use.
- **No local general chat:** install/run Ollama and pull
  `qwen2.5:3b-instruct`, or configure `PRIMUS_POD`.
- **YOLO candidate load fails:** inspect model format/runtime compatibility.
  TensorRT engine files are hardware/runtime-specific; try a compatible `.pt`.
- **JAX reports CPU fallback:** that is acceptable for a small smoke test, but
  not a GPU training validation. Check the CUDA-enabled JAX installation and
  driver on the cloud host.
- **Cloud model missing:** check `MODELS` and confirm all expected model
  directories from the download table exist.
- **MuJoCo scene/XML load error:** ensure MuJoCo is installed and use the MJX
  training scene at `sim/g1/scene_mjx.xml` for the training environment.

## 9. Local data, licenses, and safety

The project writes runtime state and downloaded/generated assets under `data/`
and model caches. This can contain personal object associations, face images,
metrics, checkpoints, or other private data. Review before sharing.

Model and robot-description licenses are independent of this repository's
license. Check the terms for each model, voice, and G1 asset before use or
redistribution.

This project is not a safety-rated controller. Simulation results, policy
exports, command routing, and a watchdog are not substitutes for validated
limits, emergency stops, collision protections, supervised testing, and
hardware-specific controls. Do not connect unvalidated policy output to
physical actuators.
