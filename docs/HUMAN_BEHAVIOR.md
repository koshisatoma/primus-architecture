# PRIMUS cognition prototypes (Sessions 1–9)

These sessions add reusable research components and a small composition layer.
They do **not** make PRIMUS human-like or turn it into a complete autonomous
robot. The helpers use simplified hand-authored dynamics and explicitly
provided inputs. The software is not safety-rated, does not infer internal
human emotions reliably, and does not execute locomotion or manipulation on
hardware.

## Session 1 — Local memory

| Component | Module | Behavior |
| --- | --- | --- |
| Salience-indexed SQLite store | `primus/memory_store.py` | Stores JSON-compatible records under `data/primus_memory.sqlite3`, supports query, decay, consolidation, and explicit forgetting. |
| Idle maintenance | `primus/replay_daemon.py` | Runs decay/consolidation in a stoppable background thread after inactivity. It does not infer facts from arbitrary sentences. |
| Optional EWC/skill adapters | `primus/anti_forgetting.py` | PyTorch utilities for training code; not connected to PPO or ASTRA optimization. |
| Optional reconstructive recall | `primus/reconstructive_recall.py` | Drift is opt-in and must not be used for factual or safety-critical inputs. |

The application records brain input/reply pairs through the integration
facade, including command and introduction turns. This is local persistent
data; clear a memory by ID through the store API or remove the database after
stopping the app. The conversational model is instructed to treat stored notes
as context rather than inventing details.

## Session 2 — Social and affect helpers

| Component | Module | Limitation |
| --- | --- | --- |
| Speech-energy turn-taking heuristic | `primus/social_cognition.py` | Needs RMS samples and people counts from a caller; it is not speaker separation. |
| Visual attention cue | `primus/social_cognition.py` | Caller supplies estimated gaze and object centers; this does not infer intent. |
| Acoustic feature extraction | `primus/social_cognition.py` | RMS/ZCR are not reliable emotion labels; valence remains neutral. |
| Bounded affect and style helpers | `primus/affect_personality.py` | Affect/personality are in-memory prototypes; no trait learning or verified TTS prosody change. |
| Time, proactive suggestion, uncertainty | `primus/proactive_agent.py` | Proactive decisions are not automatically spoken by the app. |

The current microphone implementation returns transcripts, not audio chunks or
RMS histories. Therefore audio turn-taking, prosody, and auditory attention
are library features, not currently enabled live behaviors.

## Session 3 — Spatial and proprioceptive helpers

| Component | Module | Limitation |
| --- | --- | --- |
| 3D object memory | `primus/spatial_memory.py` | Caller must supply named, metric 3D detections; existing YOLO boxes alone do not provide them. |
| Body schema and fall detector | `primus/proprioception.py` | Reads a MuJoCo model/data snapshot and requires the G1 torso body. |
| Auditory localization heuristic | `primus/proprioception.py` | Requires calibrated left/right RMS channels; it does not command a head. |
| Potential-field navigation suggestion | `primus/motor_planning.py` | Returns a bounded advisory velocity vector from a MuJoCo snapshot; it is not sent to a controller. |
| Recovery-needed state | `primus/motor_planning.py` | Does not fabricate a get-up action; raises when a fall is detected because no validated recovery policy exists. |

No depth-to-world calibration or obstacle pipeline currently feeds spatial
memory. Do not connect these suggestions to physical actuators.

## Session 4 — Composition and verification

`primus/integration.py` composes memory, style, attention, and spatial state.
`primus/brain.py` uses the facade for chat prompt context and records successful
responses. `primus/agent.py` starts/stops idle memory maintenance.

`astra_context()` currently returns a 15-value spatial vector, optionally
extended by joint values when a simulation is explicitly attached. It is not
consumed by `cloud/train_astra.py`, `cloud/wm_planner.py`, or the C++ spine.
The planner/checkpoint integration and fixed-size learned representation are
not implemented.

## Session 5 — Simulated physiology

| Component | Module | Limitation |
| --- | --- | --- |
| Sleep pressure and toy NREM/REM state | `primus/sleep_cycle.py` | Explicit helper only; no background scheduler, live listen gate, or ASTRA dream rollout. Dreams recombine explicitly stored episodic records. |
| Contact/pain and avoidance state | `primus/pain_nociception.py` | Requires caller-supplied force and contact positions; withdrawal/avoidance values are advisory and do not override motion. |
| Mood and fatigue | `primus/mood_layer.py` | Bounded toy dynamics; not a physiological measurement or validated performance predictor. |

`physiology_tick()` advances the helper states when called. The voice agent
does not automatically enter sleep, suppress listening, react to contact, or
change its runtime performance from these values.

## Session 6 — Social-developmental helpers

| Component | Module | Limitation |
| --- | --- | --- |
| Interaction-based person bond and explicit belief tracking | `primus/social_bonding.py` | In-memory state; caller must identify people and report what they observed. |
| Nonverbal symbolic suggestions | `primus/nonverbal.py` | Does not map symbolic gestures or posture offsets to robot joints. |
| Idle play suggestions | `primus/play_behavior.py` | Returns non-actuated suggestions; no manipulation or locomotion commands are sent. |

Empathy and comfort suggestions are rule-based and should not be treated as
reliable emotion recognition or mental-state inference.

## Session 7 — Cognitive-load prototypes

`primus/cognitive_load.py` provides bounded working memory, a prediction-error
comparator, simple metacognitive scores, and practice counters. The agency
comparator accepts a supplied prediction; it is not connected to a trained
ASTRA `P_lat` model. No model-based fallback or self-correction controller is
enabled.

## Session 8 — Imitation, event surprise, timing, attention, pragmatics

| Component | Module | Limitation |
| --- | --- | --- |
| Named demonstration storage and event heuristics | `primus/imitation.py` | Demonstrations are stored as arrays for offline inspection; no action retargeting or execution. |
| Indirect-request candidates | `primus/pragmatics.py` | Returns only a candidate label; it does not carry out environmental changes. |
| Cocktail-party and startle heuristics | `primus/imitation.py` | Require caller-supplied scored events; not connected to microphone processing or sound-triggered motion. |

## Developmental helpers

`primus/development.py` contains toy temperament seeds, age-window scalars,
vocabulary-stage counters, and correction tallies. These are not validated
models of human development, language acquisition, or moral reasoning.

## Session 9 — Verification status

The focused tests verify deterministic software properties of these helpers,
not human-likeness. No Human-Likeness Index, paper experiment result, ASTRA
ablation result, or hardware capability claim is reported by this repository.
The 15-value spatial state remains separate from the Session 5–8 helpers and
is not used as a grounded learned world model.

Run the focused checks:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_human_behavior.py'
python -m unittest discover -s tests -p 'test_sessions_5_9.py'
```

Run the existing voice/brain tests as well:

```bash
python -m unittest discover -s tests
```

## Demo

Run only the deterministic, non-hardware checks:

```bash
bash scripts/run_human_demo.sh
```

The script does not launch the camera, UI, or any robot-control process.
