# PRIMUS

PRIMUS is a research prototype for a voice-first, camera-aware companion and
humanoid-robot software stack. It combines a local Gradio/MuJoCo experience,
speech input and output, object/person perception, a small conversational
brain, optional cloud inference, and early ROS 2 and MJX training components.

> **Important:** This repository is a prototype, not a validated robot
> controller. The ROS 2 task executor does not implement locomotion, grasping,
> or placing, and the cloud planner is not connected to a live control loop.
> Do not connect this software to physical actuators without an independently
> validated safety and control system.

## Get started

The complete installation guide—including Python environments, system
packages, model downloads, optional cloud services, training, ROS 2, and
troubleshooting—is in **[setup.md](./setup.md)**. Start there before running
the project.

Once the local prerequisites and models are installed:

```bash
source .venv/bin/activate
python app.py
```

Open <http://127.0.0.1:7860>. For a lightweight repository smoke test:

```bash
source .venv/bin/activate
bash cloud/smoke_test.sh
```

The smoke test needs the MJX training stack (JAX, MuJoCo/MJX, NumPy, and
PyTorch). It runs small CPU-compatible training checks and does not validate
GPU serving or robot hardware.

## What the project contains

### Local application

- **Simulation and interface:** `app.py` starts the Gradio UI in
  [`primus/ui.py`](./primus/ui.py); [`primus/cams.py`](./primus/cams.py) renders
  MuJoCo scenes and telemetry. The dashboard includes three simulation views
  and RGB, depth, occupancy, gradient, and detection views of the live camera.
- **Voice:** [`primus/agent.py`](./primus/agent.py) runs the asynchronous
  listen/respond loop. [`primus/ears.py`](./primus/ears.py) uses cloud speech
  recognition when configured and falls back to local Faster Whisper;
  [`primus/mouth.py`](./primus/mouth.py) uses a local Piper voice when present
  and otherwise prints the response.
- **Brain and teaching:** [`primus/brain.py`](./primus/brain.py) routes
  conversation, introductions, object naming, text skills, and action commands.
  [`primus/objects.py`](./primus/objects.py) stores named visual associations.
  Action commands are routed as text; the standalone action-plan parser is not
  connected to a complete skill executor.
- **Cognition prototypes:** [docs/HUMAN_BEHAVIOR.md](./docs/HUMAN_BEHAVIOR.md)
  maps Sessions 1–9 and the developmental helpers to their modules, with
  explicit notes on what is integrated versus still experimental.
- **Perception and awareness:** [`primus/detect.py`](./primus/detect.py)
  performs YOLO detection; camera and person-state modules support the live
  view and social-awareness checks.
- **Scene:** [`primus/scene_builder.py`](./primus/scene_builder.py) constructs
  the MuJoCo view. [`sim/g1/`](./sim/g1/) contains the Unitree G1 MJCF assets;
  `primus/room.py` procedurally adds deterministic clutter to the scene.

### Cloud and learning components

The [`cloud/`](./cloud/) package contains:

- MJX reinforcement-learning training in `train_reflex_ppo.py`.
- Simulation/target episode collection in `collect_episodes.py`.
- ASTRA-style source grounding and target-encoder alignment in
  `train_astra.py`.
- Policy export for the C++ spine in `export_policy.py`.
- Local MuJoCo teacher demonstrations and policy-correction refinement in
  `teleop.py` and `train_from_teacher.py`.
- Optional vLLM, Moshi, transcription, depth, and person-embedding services.
- A latent-space CEM planner prototype in `wm_planner.py`.

The planner is not currently integrated into the application's agent, the ROS 2
executor, or the C++ control loop. The current ASTRA trainer writes a checkpoint
state dictionary; the standalone planner currently expects an executable model
object exposing `plat` and `rpred`. Treat the planner as experimental until
checkpoint loading and runtime integration are completed and tested.

### C++ and ROS 2

- `core_cpp/` contains camera, reflex, and spine source files; locally built
  executables are not source-controlled.
- [`primus_ros2/`](./primus_ros2/) contains a ROS 2 Humble package with camera/
  perception, brain, and task-executor bridges. Its executor reports that a
  motion backend is unavailable rather than claiming a robot action ran.
- [`safety_rs/`](./safety_rs/) is an early Rust crate; it currently has no
  implementation dependencies or completed safety controller.

For ROS package-specific notes, see
[`primus_ros2/README.md`](./primus_ros2/README.md). The upstream G1 model notes
are in [`sim/g1/README.md`](./sim/g1/README.md).

## Model inventory

Model weights and generated training artifacts are local assets and are not
included in Git. The root `.gitignore` excludes `data/`, model weights, ONNX
files, and TensorRT engines.

| Use | Model / asset | How it is used |
| --- | --- | --- |
| Local object detection | YOLO26x/YOLO11x or YOLO11n `.pt`/`.engine` candidates | `primus.detect` searches in priority order; see [`setup.md`](./setup.md). |
| Optional segmentation | `yolo26n-seg.pt` | Present in some workspaces; not currently selected by `primus.detect`. |
| Local speech recognition | Faster Whisper `base` | Downloaded by Faster Whisper on first use when no cloud transcription is available. |
| Local speech output | Piper `en_US-hfc_female-medium.onnx` and its JSON sidecar | Loaded from `data/`; response text is printed if unavailable. |
| Local chat | Ollama `qwen2.5:3b-instruct` | Used by the local conversation path. |
| Cloud chat | Qwen2.5-VL-7B-Instruct AWQ | Downloaded by `cloud/download_models.sh`; served through vLLM. |
| Cloud speech | Kyutai Moshi; Faster Whisper `large-v3` | Moshi is launched as a separate service; large-v3 serves `/transcribe`. |
| Cloud perception | Depth Anything V2 Large; DINOv2 Large; YOLO detector | Used by the optional depth and person-embedding endpoints. |
| Learned policies / world model | PPO checkpoints, ASTRA checkpoint, exported policy weights | Produced by the training/export commands in [`cloud/README.md`](./cloud/README.md). |

The repository may contain different downloaded YOLO formats on a developer
machine. For the exact candidate precedence and supported download commands,
read the setup guide; files in the workspace snapshot are not necessarily
tracked or installed automatically.

## Data and privacy

Runtime data may include object names, face images, metrics, model outputs, and
learned skills under `data/`. It can include personal or locally collected
information. Review it before sharing, uploading, or committing it. Do not
commit credentials or private model access tokens.

## Project status

This is active prototype software. A parser or training task being present does
not mean the complete skill is executable on a physical G1. Verify dependencies,
model licenses, interfaces, and hardware safety independently before deployment.
