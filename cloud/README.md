# PRIMUS cloud and training components

This folder contains optional model-serving and learning workflows. The
repository-wide install, model inventory, and machine setup are documented in
[`../setup.md`](../setup.md). This README describes what the scripts here do
and how to run the workflows after setup.

## Cloud services

`cloud/serve_all.sh` starts three processes:

| Port | Service | Implementation |
| --- | --- | --- |
| 8000 | OpenAI-compatible chat API | vLLM serving the local `qwen-vl-7b` model directory |
| 8001 | Voice service | `moshi.server` |
| 8002 | Perception API | `cloud.svc_percept` (transcription, depth estimation, person embeddings) |

On a prepared GPU host:

```bash
export MODELS="$HOME/models"
export PRIMUS_VENV="$HOME/cx"
bash cloud/serve_all.sh
```

Point a local PRIMUS process at the host's externally reachable base URL:

```bash
export PRIMUS_POD="http://<private-host-or-tunnel-address>"
```

The client uses port 8000 for chat and port 8002 for speech transcription.
Keep these unauthenticated services on a trusted private network; do not expose
them directly to the public internet. See [`../setup.md`](../setup.md) for model
downloads, service dependency installation, and the current endpoint details.

## Training and export

The training workflows need JAX, MuJoCo, MuJoCo-MJX, NumPy, and PyTorch. Start
with a short smoke run before using long training schedules:

```bash
source .venv/bin/activate
bash cloud/smoke_test.sh
```

Example small PPO run:

```bash
python -m cloud.train_reflex_ppo \
  --task stand --envs 2 --steps 16 --T 8 --out data/ckpt
```

The output prefix `data/ckpt` produces a task-suffixed checkpoint, for example
`data/ckpt_stand.npz`. Collect simulation episodes and train the ASTRA model:

```bash
python -m cloud.collect_episodes --eps 2 --T 8 --out data/episodes
python -m cloud.train_astra \
  --data data/episodes.npz --epochs 1 --align_epochs 1 \
  --out data/astra_grounding.pt
```

Export a trained policy for use by the C++ spine:

```bash
python -m cloud.export_policy \
  --ckpt data/ckpt_stand.npz --out data/stand_policy
```

This writes `data/stand_policy_weights.npz` and the flat layer files plus
`policy_spec.json` under `data/stand_policy_cpp/`. Add `--onnx` only when
`jax2onnx` is installed and ONNX output is needed.

For a GPU training pod, [`../scripts/cloud_bootstrap.sh`](../scripts/cloud_bootstrap.sh)
creates a Python virtual environment with JAX CUDA, MuJoCo/MJX, NumPy, PyTorch,
FastAPI, and Uvicorn. It also clones or fast-forwards the repository. Override
its paths using `PRIMUS_WORKDIR`, `PRIMUS_REPO_DIR`, and `PRIMUS_VENV`. The
bootstrap covers the training base; cloud serving additionally needs compatible
vLLM, Moshi, Transformers, Faster Whisper, OpenCV, and Ultralytics versions.

## Current limitations

- The small smoke test does not establish full-scale training convergence.
- Task-specific target/object arguments are not exposed by the PPO CLI. In
  particular, reach/grasp/lift/place training should not be treated as
  task-grounded merely because those task names are accepted.
- `wm_planner.py` is a standalone prototype, not wired into the voice agent,
  ROS executor, or C++ control loop. Its current loader expects a model object
  implementing `plat` and `rpred`, while `train_astra.py` saves a checkpoint
  dictionary; an adapter and real-time integration are still required.
- The ROS 2 executor has no locomotion or manipulation backend. A trained
  policy export is not itself a validated hardware controller.

Treat training artifacts as generated, machine-specific data. The `data/`
directory is ignored by Git and may contain personal or sensitive runtime data.
