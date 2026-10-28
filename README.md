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

After installing the local prerequisites and models, follow the run
instructions in [setup.md](./setup.md) to start the application. The same
guide documents the optional smoke test, which needs the MJX training stack
(JAX, MuJoCo/MJX, NumPy, and PyTorch) and does not validate GPU serving or
robot hardware.

## What the project contains

### Local application

- **Simulation and interface:** `app.py` starts the Gradio UI in
  [`primus/ui.py`](./primus/ui.py); [`primus/cams.py`](./primus/cams.py) renders
  MuJoCo scenes and telemetry.
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

## Research direction and current scope

PRIMUS explores how conversational interaction, perception, simulation,
learning, and memory-related helpers might fit together in a robot-oriented
software stack. The repository includes prototypes for salience-based local
memory, affect and social cues, spatial state, cognitive load, imitation, and
developmental counters. Their current implementation and limitations are
summarized in [docs/HUMAN_BEHAVIOR.md](./docs/HUMAN_BEHAVIOR.md).

These are research components, not evidence that the system learns like a
person or reliably infers human emotions, beliefs, or intent. Many helpers
require explicit inputs from a caller and are not connected to the live
camera, microphone, learned policy, or robot controller. The project does not
report a validated human-likeness score or demonstrate safe physical
locomotion or manipulation.

The design draws on ideas from developmental psychology, cognitive science,
affective science, and motor learning. These are inspirations for software
experiments, not claims that the system reproduces human development,
consciousness, emotion, or cognition. The modules use simplified rules and
explicitly supplied state; their implemented behavior and limitations are
described in the [cognition prototype guide](./docs/HUMAN_BEHAVIOR.md).

## Architecture at a glance

The current software can be understood as several connected and experimental
parts:

| Area | Current role |
| --- | --- |
| Interaction | Gradio application, asynchronous voice loop, conversation routing, and local or optional cloud model calls. |
| Perception | Camera capture and YOLO detections support the live view and object/person-related features. |
| Simulation | MuJoCo renders a G1 scene for the application; MJX provides a separate environment for policy experiments. |
| Learning | JAX-based PPO training, episode collection, ASTRA model training, and policy export are available as separate workflows. |
| Cognition prototypes | Local memory and explicitly called social, affective, spatial, developmental, and cognitive-load helpers. |
| Integration experiments | C++ camera/reflex/spine sources and ROS 2 bridges; not a complete hardware control system. |

These components are not one end-to-end autonomous pipeline. In particular,
the planning prototype is not connected to the voice agent or robot control
loop, and most cognition helpers are not automatically driven by live sensor
data.

## Memory and cognitive prototypes

The memory store in [`primus/memory_store.py`](./primus/memory_store.py)
persists JSON-compatible records in a local SQLite database. Its API supports
salience-aware queries, decay, consolidation, and explicit forgetting.
[`primus/replay_daemon.py`](./primus/replay_daemon.py) can perform idle
maintenance in a stoppable background thread. These mechanisms do not extract
reliable facts from arbitrary conversation or implement a complete human
episodic/semantic/procedural memory system.

Other modules explore reconstructive recall, optional anti-forgetting
utilities, bounded working memory, prediction-error comparisons, practice
counters, and proactive suggestion rules. They remain helper libraries or
training utilities; for example, the EWC/adapter utilities are not integrated
into PPO or ASTRA training, and the prediction comparator does not consume a
trained ASTRA dynamics model.

## Social, affective, and developmental prototypes

Modules including [`primus/social_cognition.py`](./primus/social_cognition.py),
[`primus/affect_personality.py`](./primus/affect_personality.py),
[`primus/social_bonding.py`](./primus/social_bonding.py),
[`primus/pragmatics.py`](./primus/pragmatics.py), and
[`primus/development.py`](./primus/development.py) provide small, explicit
state updates and heuristic suggestions. Some require caller-supplied values
such as audio energy, gaze estimates, identified people, observations, or
practice events. The live microphone path supplies transcripts rather than the
audio histories these helpers would need for turn-taking or acoustic analysis.
These features should not be read as validated emotion recognition, theory of
mind, attachment, language development, or autonomous behavior.

## Simulation, policy learning, and ASTRA

The MJX environment in [`cloud/g1_env.py`](./cloud/g1_env.py) defines
simulation tasks for standing, walking, sitting, reaching, grasping, lifting,
and placing. The PPO trainer learns policies in simulation; task names do not
by themselves mean the task is grounded to a real target. In particular,
target/object configuration and meaningful validation are still required for
the manipulation reward paths.

[`cloud/train_astra.py`](./cloud/train_astra.py) trains a recurrent source
encoder with transition, reward, and absolute-state prediction heads, then
aligns a target observation encoder using maximum mean discrepancy. The
episode collector currently records stand-policy simulation experience; the
workflow is not a demonstrated real-world teacher-data pipeline or verified
sim-to-real transfer. The ASTRA checkpoint is a state dictionary, while
[`cloud/wm_planner.py`](./cloud/wm_planner.py) currently expects a model object
with `plat` and `rpred` methods. The planner and trainer therefore need an
adapter and integration before they can be used together.

The policy exporter creates weights and C++-oriented artifacts, but this is
not proof of a real-time control guarantee or a safe physical policy. For
available commands and training caveats, see the
[cloud guide](./cloud/README.md).

## Perception, ROS 2, and safety scope

The local detector in [`primus/detect.py`](./primus/detect.py) selects from
supported YOLO `.pt` and TensorRT engine candidates in the repository root.
The cloud perception service separately exposes transcription, depth, and
person-embedding endpoints; depth and re-identification are not currently a
complete application perception pipeline. Model formats and machine-specific
weights are covered in [setup.md](./setup.md).

The ROS 2 package provides camera/perception, brain, and task-executor
bridges. The task executor has no locomotion or manipulation backend. The Rust
crate in [`safety_rs/`](./safety_rs/) is an early watchdog prototype, not a
safety-rated controller or substitute for hardware emergency stops, validated
limits, collision protections, and supervised testing.

## Evaluation status

Repository tests cover deterministic software behavior, imports, and small
training workflows. They do not measure human-likeness, establish model
convergence at full scale, demonstrate sim-to-real transfer, or validate
hardware behavior. The smoke-test workflow and its prerequisites are described
in [setup.md](./setup.md); detailed cognition test scope is in
[docs/HUMAN_BEHAVIOR.md](./docs/HUMAN_BEHAVIOR.md).

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
