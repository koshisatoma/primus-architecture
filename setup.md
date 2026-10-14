# Setup

This document covers the current Linux development setup for Primus. The
workspace does not yet have a root Python dependency manifest, so the checked-in
`.venv` is the only existing snapshot of the Python environment; a clean
installation cannot currently be reproduced from a single `pip install`
command.

## Main application

### Prerequisites

- Linux with Python 3.10 (the workspace's existing virtual environment uses
  Python 3.10).
- A camera and microphone for live teacher-camera and voice input.
- The Python packages used by the app: Gradio, MuJoCo, NumPy, OpenCV,
  Ultralytics, Faster Whisper, SoundDevice, Piper TTS, and Requests.
- At least one compatible YOLO model file in the workspace root. The detector
  searches for `yolo26x.engine`, `yolo26x.pt`, `yolo11x.engine`, `yolo11x.pt`,
  `yolo11n.engine`, then `yolo11n.pt`, and uses the first loadable model.
  TensorRT engine files require a compatible NVIDIA/TensorRT environment;
  PyTorch weights are the alternative when available.
- A local Ollama service with the `qwen2.5:3b-instruct` model for general
  conversation. The app sends chat requests to
  `http://127.0.0.1:11434/api/chat`. Without Ollama, only its small fixed
  greeting/distress/error replies are available.

The repository currently contains a `.venv`. To use that existing environment:

```bash
cd /path/to/primus3
source .venv/bin/activate
```

If creating a fresh environment, install the prerequisites above using
versions compatible with your platform and hardware. There is no checked-in
`requirements.txt` or root `pyproject.toml` to pin those versions yet.

### Run

```bash
cd /path/to/primus3
source .venv/bin/activate
python app.py
```

Open <http://127.0.0.1:7860>. The app uses MuJoCo to render the simulated G1
scene and reads the teacher camera from the C++ shared-memory camera when
available; otherwise it tries OpenCV camera index 0 directly.

The optional C++ camera producer is the existing `cam_bin` executable. Start
it in a separate terminal before the Python app if you want the shared-memory
camera path:

```bash
cd /path/to/primus3
./cam_bin
```

The producer needs access to camera index 0 and creates shared memory named
`primus_cam`. `cam_bin` is a locally built executable and is ignored by the
root `.gitignore`; its source is in `core_cpp/cam.cpp`. The workspace does not
currently include a CMake file or a documented compile command for rebuilding
that executable.

Voice input uses the system audio devices. Whisper may need to download its
`base` model on first use. Piper speech output loads
`data/en_US-hfc_female-medium.onnx`; if that voice file or Piper is unavailable,
replies are printed instead of spoken.

## Optional cloud inference

The optional `cloud/` package provides MJX PPO training, ASTRA episode
collection/training, policy export, model downloads, and pod inference services.
Model downloads need internet access and disk space, but do not use a GPU:

```bash
cd /path/to/primus3
bash cloud/download_models.sh
```

The default download directory is `$HOME/models`; pass a directory as the first
argument or set `MODELS` to override it. The Moshi model may require accepting
its Hugging Face terms before it can be downloaded. Use
`bash scripts/cloud_bootstrap.sh` on a compatible GPU pod to create the cloud
training environment and clone this repository. It uses `$HOME/cx` by default;
set `PRIMUS_WORKDIR`, `PRIMUS_REPO_DIR`, or `PRIMUS_VENV` to override those
locations. Activate the same environment to download and serve models.

The local client uses the cloud brain and transcription service when
`PRIMUS_POD` is set, and falls back to local Ollama and Whisper when the cloud
request fails:

```bash
export PRIMUS_POD=http://<pod-address>:8000
```

On the pod, install the serving stack in its Python environment, download the
models, then launch the services with `bash cloud/serve_all.sh`. The brain uses
port 8000, Moshi uses 8001, and transcription/depth/person re-identification
use 8002. Serving requires the appropriate GPU drivers and model-compatible
versions of vLLM, Moshi, PyTorch, Transformers, Faster Whisper, OpenCV, and
Ultralytics. The downloads and service launch have not been tested against a
live GPU pod.

Run `bash cloud/smoke_test.sh` locally to check Python syntax and the local
cloud-client fallback paths without loading GPU models. It does not validate
the training stack or exercise a live pod.

`cloud/smoke_test.sh` runs the unit tests and a small end-to-end CPU-capable
training/export smoke. The training stack requires JAX, `mujoco-mjx`, MuJoCo,
NumPy, and PyTorch in addition to the application dependencies. The environment
does not need a GPU for this small smoke, although full PPO training benefits
from one. The MJX training environment uses
`sim/g1/scene_mjx.xml`; `sim/g1/scene.xml` currently fails MuJoCo loading
because its eye camera targets a missing body.

The service processes bind to all interfaces by default. They do not implement
authentication; restrict exposed pod ports to trusted clients or put them
behind an authenticated private gateway. Do not expose these endpoints
directly to the public internet.

## ROS 2 (optional)

The ROS 2 package targets ROS 2 Humble and requires `colcon`. Build it from its
own directory:

```bash
cd /path/to/primus3/primus_ros2
source /opt/ros/humble/setup.bash
colcon build --packages-select primus_actions
source install/setup.bash
```

For the installed perception node, point it to the workspace root:

```bash
export PRIMUS3_ROOT=/path/to/primus3
```

Start the camera producer first if the shared-memory camera is desired, then
run each node in a separate terminal (with the ROS and workspace environment
sourced):

```bash
ros2 run primus_actions perception_bridge
ros2 run primus_actions task_executor
ros2 run primus_actions brain_bridge
```

To send a test voice-text message:

```bash
ros2 topic pub --once /primus/voice_text std_msgs/msg/String \
  "{data: 'find the cup'}"
```

The nodes publish perception data and route/classify commands, but the task
executor has no locomotion, navigation, grasping, or placing backend. It
reports that no motion was executed. The main app also sends recognized action
commands directly to `/primus/command`; `brain_bridge` separately relays
`/primus/voice_text` to that topic.

## Tests

Run the current Python unit tests from the workspace root:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

## Local data and safety

Names/face crops, object associations, taught text skills, and metrics are
stored under `data/`. These files can contain personal or locally collected
data; review them before sharing or committing them.

This project is a prototype. The simulation, command routing, and standalone
watchdog are not a substitute for a validated hardware safety system. Do not
connect the command path to physical actuators without implementing and testing
appropriate control and safety backends.
