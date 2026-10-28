# PRIMUS: Tabula Rasa Human-Like Robotics Architecture

![Status](https://img.shields.io/badge/status-active_development-brightgreen)
![License](https://img.shields.io/badge/license-Apache_2.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![C++](https://img.shields.io/badge/C++-20-blue)
![Rust](https://img.shields.io/badge/Rust-1.70+-orange)

**PRIMUS** (Primary Recursive Intelligent Multidimensional Unified System) is a comprehensive architectural framework for building genuinely human-like robotic agents. Unlike traditional robotics systems pre-loaded with vast datasets and pre-trained models, PRIMUS begins as a true **tabula rasa** — a blank slate with zero prior task knowledge. It learns locomotion, manipulation, social interaction, and cognitive behaviors entirely through real-time teaching, ASTRA-grounded simulation, and a biologically inspired, selectively forgettable memory system.

This repository contains the complete software stack: multi-modal perception pipelines, advanced cognitive and social modules, MuJoCo/MJX simulation environments, ASTRA world-model training scripts, a 500Hz C++ reflex spine, Rust-based safety watchdog, ROS 2 integration, and a dual-column Gradio vision dashboard.

---

## Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Architecture Overview](#3-architecture-overview)
4. [System Modules](#4-system-modules)
5. [The Human-Likeness Battery](#5-the-human-likeness-battery)
6. [ASTRA Grounding Framework](#6-astra-grounding-framework)
7. [Memory Architecture](#7-memory-architecture)
8. [Cognitive & Social Systems](#8-cognitive--social-systems)
9. [Motor Control & Learning](#9-motor-control--learning)
10. [Perception Pipeline](#10-perception-pipeline)
11. [Safety & Engineering](#11-safety--engineering)
12. [Developmental Trajectory](#12-developmental-trajectory)
13. [Research Context](#13-research-context)
14. [Citation](#14-citation)

---

## 1. Core Philosophy

PRIMUS is built on three foundational pillars that distinguish it from conventional robotics architectures:

### 1.1 Tabula Rasa (Zero-Prior Learning)
The agent starts with **no pre-programmed skills**. When asked to perform an untaught task, it explicitly refuses and requests instruction. This is not a limitation but a design principle: PRIMUS learns exactly like a human child — through demonstration, verbal instruction, and guided practice.

### 1.2 Selective Memory (Forgettable, but Not Catastrophic)
Human memory is not a perfect recording device. We forget trivial details exponentially (Ebbinghaus forgetting curve) while retaining emotionally salient or frequently rehearsed events for life. PRIMUS replicates this through:
- **Salience-weighted decay**: Each memory is tagged with emotional valence, novelty, and access frequency. Low-salience traces decay via $S(t) = S_0 \cdot e^{-kt}$.
- **Sleep-phase consolidation**: During idle periods, the `primus/replay_daemon.py` promotes high-salience episodic memories to semantic/procedural stores.
- **Anti-catastrophic forgetting**: Using Elastic Weight Consolidation (EWC) and per-skill LoRA adapters, learning a new primitive never overwrites a previously learned skill.

### 1.3 ASTRA-Grounded Sim2Real
Traditional sim-to-real transfer relies on domain randomization. PRIMUS uses the **ASTRA** (Approximate Information State) framework, which grounds an abstract simulator to reality through history-based corrections. The loss function combines three terms:
$$\mathcal{L} = \lambda_1 \mathcal{L}_{trans} + \lambda_2 \mathcal{L}_{rew} + \lambda_3 \mathcal{L}_{abs}$$
Where $\mathcal{L}_{trans}$ is latent transition error, $\mathcal{L}_{rew}$ is reward prediction error, and $\mathcal{L}_{abs}$ is abstraction error matching real-world teacher demonstrations.

---

## 2. Theoretical Foundations
PRIMUS integrates insights from multiple disciplines:
- **Developmental Psychology**: Bowlby's Attachment Theory, Piaget's Sensorimotor Stages, Vygotsky's Zone of Proximal Development.
- **Cognitive Neuroscience**: Cowan's Working Memory Model (~4 chunks), Fitts-Posner Skill Acquisition, Blakemore's Forward Model (sense of agency).
- **Affective Science**: Russell's Circumplex Model (valence-arousal), Yerkes-Dodson Law (arousal vs. performance).
- **Social Cognition**: Theory of Mind (false-belief tracking), Joint Attention, Emotional Contagion.
- **Motor Control**: Bernstein's Degrees of Freedom Problem, Schmidt's Schema Theory, Dynamic Systems Theory.

---

## 3. Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    PERCEPTION LAYER                         │
│  RGB-D sensing | Object detection | Depth estimation | Re-ID│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 COGNITION & SOCIAL LAYER                    │
│  Intent routing | ToM | Emotion | Personality | Proactivity │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY LAYER                           │
│  5-tier system | Ebbinghaus decay | Sleep consolidation     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  MOTOR & CONTROL LAYER                      │
│  500Hz spine | MJX PPO | ASTRA grounding | Teleop teaching  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    SAFETY LAYER                             │
│  UDP watchdog | E-stop | Health monitoring (Rust)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. System Modules

### 4.1 Perception Layer
- **`primus/shmcam.py` & `core_cpp/cam.cpp`**: 1080p RGB capture via V4L2 → POSIX shared memory ring buffer (<10ms latency).
- **`primus/detect.py`**: YOLO26x TensorRT FP16 inference (~9ms per frame), person-first ordering.
- **`cloud/svc_percept.py`**: FastAPI service providing Whisper-large-v3 STT, Depth-Anything-V2 metric depth, and DINOv2 person re-identification.
- **`primus/objects.py` & `primus/people.py`**: Touch-to-name symbol grounding and left-to-right face-name binding.

### 4.2 Cognition & Social Layer
- **`primus/brain.py`**: Intent router, zero-prior gate, skill recall, action dispatch.
- **`primus/social.py`**: Ollama/vLLM chat interface with dynamic personality injection.
- **`primus/social_cognition.py`**: Audio turn-taking (VAD), Theory of Mind (joint attention), emotion recognition (prosody).
- **`primus/affect_personality.py`**: VAD emotion system, Big Five personality drift, humor and register control.
- **`primus/proactive_agent.py`**: Time sense, proactive initiation ("Remember when…"), self-awareness (calibrated uncertainty).
- **`primus/social_bonding.py`**: Bowlby attachment, empathy, false-belief ToM.

### 4.3 Memory Layer
- **`primus/memory_store.py`**: SQLite backend with salience indexing and Ebbinghaus exponential decay.
- **`primus/replay_daemon.py`**: Idle-time consolidation (hippocampal-neocortical replay).
- **`primus/anti_forgetting.py`**: Per-skill LoRA adapters and Elastic Weight Consolidation (EWC) penalty.
- **`primus/reconstructive_recall.py`**: Memories reconstructed with mild Gaussian drift (human-like misremembering).

### 4.4 Motor & Control Layer
- **`cloud/g1_env.py`**: MJX (JAX-accelerated MuJoCo) G1 environment with abstraction function $\phi(s)$.
- **`cloud/train_reflex_ppo.py`**: Pure-JAX PPO implementation parallelized across 2048-4096 environments.
- **`cloud/collect_episodes.py`**: Golemo protocol for collecting paired sim/real transitions.
- **`cloud/train_astra.py`**: ASTRA Algorithms 1 & 2 (Grounding + MMD target alignment).
- **`cloud/export_policy.py`**: Exports JAX policy weights to flat `.bin` files for C++ spine.
- **`core_cpp/spine.cpp`**: 500Hz control loop loading `.bin` weights.
- **`cloud/teleop.py`**: HG-DAgger style teleoperation for teacher demonstrations and corrections.

### 4.5 Safety Layer
- **`safety_rs/src/main.rs`**: Independent Rust process, UDP listener on port 9119, hard E-stop capability.

### 4.6 Cloud/Simulation Layer
- **`primus/viewops.py`**: 5-mode live camera processing (RGB, Depth, Occupancy, Gradient, Detection).
- **`primus/ui.py`**: Gradio dual-column layout (MuJoCo sim left, live camera right).
- **`primus/render.py`**: Offscreen MuJoCo renderer (kitchen, front3m, corner views).

---

## 5. The Human-Likeness Battery

PRIMUS is evaluated against an 18-dimensional behavioral battery:
1. **Ebbinghaus Decay + Salience Ordering**: Trivial memories fade; important ones persist.
2. **Replay Consolidation**: Idle-time "sleep" promotes episodic to semantic memory.
3. **Reconstructive Recall**: Memories retrieved with mild, human-like drift.
4. **Audio Turn-Taking**: Stays silent when humans converse with each other.
5. **VAD Emotion → Prosody**: Arousal/valence dynamically modulate TTS pitch and rate.
6. **Object Permanence**: 3D spatial map persists when objects leave the camera frame.
7. **Sleep Pressure + Onset**: Unresponsive during sleep; wakes with inertia.
8. **Nociception + Hyperalgesia**: Pain triggers withdrawal reflex and lasting avoidance.
9. **Attachment**: Forms bonds with primary caregivers; exhibits stranger anxiety.
10. **False-Belief ToM**: Understands when a teacher holds an incorrect belief about object location.
11. **Cowan WM Capacity**: Limited to ~4 chunks; dual-tasking causes measurable interference.
12. **Fitts-Posner Automaticity**: Practiced skills become faster and free up working memory.
13. **Violation-of-Expectation**: High latent prediction error (ASTRA) triggers surprise.
14. **Startle Sensitization**: Strong events temporarily amplify reflex responses.
15. **Cocktail-Party Attention**: Own name or teacher's voice captures auditory focus.
16. **Implicature**: Understands indirect requests ("It's cold in here" → close window).
17. **Language Stage Transitions**: Vocabulary growth follows babbling → one-word → two-word.
18. **Sense of Agency**: Forward-model comparator distinguishes self-caused actions from external events.

*Run `python tests/test_human_likeness_battery.py` to generate the Human-Likeness Index (HLI) score.*

---

## 6. ASTRA Grounding Framework

Instead of making the simulator photorealistic, ASTRA trains a self-predictive latent dynamics model that learns to predict its own errors using history-based corrections.

**Loss Function (Algorithm 1)**:
$$\mathcal{L} = \lambda_1 \mathcal{L}_{trans} + \lambda_2 \mathcal{L}_{rew} + \lambda_3 \mathcal{L}_{abs}$$

**Target Alignment (Algorithm 2)**: After training the source encoder $\psi_s$, we freeze it and train a target encoder $\psi_t$ via Maximum Mean Discrepancy (MMD):
$$\mathcal{L}_{align} = \text{MMD}(\psi_s(h_t^{sim}), \psi_t(h_t^{real}))$$

This aligns the source (sim) and target (real) latent spaces, enabling zero-shot transfer of policies trained in the grounded simulator.

---

## 7. Memory Architecture

PRIMUS implements a **5-tier memory system**:
1. **Procedural**: Taught procedures (DAGs of primitives). Permanent.
2. **Episodic**: Time-stamped interaction episodes. Decays exponentially unless consolidated.
3. **Semantic**: Facts extracted from episodes. Permanent.
4. **Sensory**: Raw perceptual traces (circular buffer, ~1 second).
5. **Working**: Active chunks under attentional control (~4 chunks, ~30 seconds).

**Forgetting Mechanisms**:
- Ebbinghaus Decay: $S(t) = S_0 \cdot e^{-kt}$
- Interference Forgetting: Conflicting memories overwrite based on similarity.
- Reconstructive Drift: Numeric fields perturbed by $\mathcal{N}(0, 0.05^2)$ on recall.

---

## 8. Cognitive & Social Systems

- **Theory of Mind (ToM)**: Tracks what others know vs. what PRIMUS knows (false-belief task competence).
- **Emotional Contagion**: Catches and mirrors the emotional state of interaction partners ($v_{primus} = \beta \cdot v_{human}$).
- **Personality Drift**: Big Five traits (OCEAN) drift slowly based on experience ($\alpha = 0.01$).
- **Proactive Initiation**: After 30+ minutes of idle time, PRIMUS initiates conversation by recalling high-salience episodic memories.

---

## 9. Motor Control & Learning

**Primitive Library**: 8 core motor primitives trained via PPO in MJX (stand, walk, walk_to, sit, reach, grasp, lift, place).
**PPO Implementation**: Pure-JAX, 2048-4096 parallel envs, T=256 horizon, GAE $\lambda=0.95$, clip range [0.8, 1.2].
**C++ Spine Integration**: Exported policies run at 500Hz on the real robot, loading `.bin` weights directly without Python dependencies.

---

## 10. Perception Pipeline

- **Camera Capture**: V4L2 → POSIX shared memory ring buffer (<10ms latency).
- **Object Detection**: YOLO26x TensorRT FP16 (~9ms inference).
- **Depth Estimation**: Depth-Anything-V2 metric depth from monocular RGB.
- **Person Re-ID**: DINOv2 embeddings for face-name binding.

---

## 11. Safety & Engineering

- **Rust Watchdog**: Independent process monitoring heartbeat via UDP (port 9119). Hard E-stop kills process tree on timeout.
- **ROS 2 Integration**: Perception and brain bridges for external planners.

---

## 12. Developmental Trajectory

PRIMUS follows a developmental sequence analogous to human child development:
- **0-10h**: Sensorimotor Reflexes (basic reflexes, no intentional action)
- **10-50h**: Primary Circular Reactions (repeated actions for sensory feedback)
- **50-200h**: Secondary Circular Reactions (object permanence, means-end reasoning)
- **200-500h**: Coordination of Schemes (intentional action sequences)
- **500-1000h**: Tertiary Circular Reactions (experimentation, trial-and-error)
- **1000h+**: Symbolic Thought (language acquisition, abstract reasoning)

---

## 13. Research Context

PRIMUS contributes to:
1. **Tabula Rasa Robotics**: Learning from scratch without pre-training.
2. **Sim2Real Transfer**: ASTRA framework for grounding abstract simulators.
3. **Human-Like AI**: Biologically inspired memory, emotion, and social cognition.


*Built with ❤️ for the future of human-robot collaboration.*