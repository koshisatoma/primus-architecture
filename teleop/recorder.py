"""Record keyboard-guided demonstrations with the fixed-base Unitree G1."""

import argparse
import math
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from teleop.teacher import (
    PROJECT_ROOT,
    SKILLS_DIRECTORY,
    register_learned_skill,
    task_filename,
)


DEFAULT_SCENE = PROJECT_ROOT / "sim" / "g1" / "scene_teleop.xml"
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
RIGHT_HAND_JOINTS = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_SPACE = 32
JOINT_STEP = 0.05


def _model_path(path: str | Path) -> Path:
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path


def _named_ids(
    model: mujoco.MjModel,
    names: tuple[str, ...],
    object_type: mujoco.mjtObj,
) -> list[int]:
    ids = [mujoco.mj_name2id(model, object_type, name) for name in names]
    missing = [name for name, object_id in zip(names, ids) if object_id < 0]
    if missing:
        raise ValueError(f"G1 scene is missing required names: {', '.join(missing)}")
    return ids


def _set_fixed_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_pose: np.ndarray,
) -> None:
    data.qpos[:7] = base_pose
    data.qvel[:6] = 0
    mujoco.mj_forward(model, data)


def record_task(
    task_name: str,
    xml_path: str | Path = DEFAULT_SCENE,
    record_hz: float = 30.0,
) -> Path | None:
    if not math.isfinite(record_hz) or record_hz <= 0:
        raise ValueError("Recording frequency must be greater than zero")
    filename = task_filename(task_name)

    model = mujoco.MjModel.from_xml_path(str(_model_path(xml_path)))
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if keyframe_id < 0:
        raise ValueError("G1 scene must define a 'stand' keyframe")

    _named_ids(model, RIGHT_ARM_JOINTS, mujoco.mjtObj.mjOBJ_JOINT)
    arm_actuator_ids = _named_ids(
        model, RIGHT_ARM_JOINTS, mujoco.mjtObj.mjOBJ_ACTUATOR
    )
    hand_joint_ids = _named_ids(
        model, RIGHT_HAND_JOINTS, mujoco.mjtObj.mjOBJ_JOINT
    )
    hand_actuator_ids = _named_ids(
        model, RIGHT_HAND_JOINTS, mujoco.mjtObj.mjOBJ_ACTUATOR
    )
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected the G1 scene to use a floating base joint")

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    base_pose = data.qpos[:7].copy()
    initial_controls = data.ctrl.copy()
    arm_targets = np.array(
        [data.ctrl[actuator_id] for actuator_id in arm_actuator_ids],
        dtype=np.float64,
    )
    open_hand_targets = np.array(
        [data.ctrl[actuator_id] for actuator_id in hand_actuator_ids],
        dtype=np.float64,
    )
    closed_hand_targets = open_hand_targets.copy()
    for index, actuator_id in enumerate(hand_actuator_ids):
        lower, upper = model.actuator_ctrlrange[actuator_id]
        closed_hand_targets[index] = (
            lower
            if abs(open_hand_targets[index] - lower)
            > abs(open_hand_targets[index] - upper)
            else upper
        )

    lock = threading.Lock()
    selected_joint = [0]
    recording_requested = [False]
    save_requested = [False]
    close_hand_requested = [False]
    open_hand_requested = [False]
    frames: list[np.ndarray] = []
    recording = False
    hand_closed = False
    next_record_time = 0.0
    should_save = False

    def on_key(key: int) -> None:
        character = chr(key).lower() if 32 <= key < 128 else ""
        with lock:
            if character in {"1", "2", "3", "4", "5", "6", "7"}:
                selected_joint[0] = int(character) - 1
            elif key in (KEY_UP, KEY_RIGHT):
                selected_joint_index = selected_joint[0]
                actuator_id = arm_actuator_ids[selected_joint_index]
                lower, upper = model.actuator_ctrlrange[actuator_id]
                arm_targets[selected_joint_index] = np.clip(
                    arm_targets[selected_joint_index] + JOINT_STEP,
                    lower,
                    upper,
                )
            elif key in (KEY_DOWN, KEY_LEFT):
                selected_joint_index = selected_joint[0]
                actuator_id = arm_actuator_ids[selected_joint_index]
                lower, upper = model.actuator_ctrlrange[actuator_id]
                arm_targets[selected_joint_index] = np.clip(
                    arm_targets[selected_joint_index] - JOINT_STEP,
                    lower,
                    upper,
                )
            elif key == KEY_SPACE:
                recording_requested[0] = not recording_requested[0]
            elif character == "g":
                close_hand_requested[0] = True
            elif character == "f":
                open_hand_requested[0] = True
            elif character == "s":
                save_requested[0] = True

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    print(f"=== RECORDING TASK: {task_name} (Unitree G1 right arm) ===")
    print("1-7: select right-arm joint | arrows: move selected joint")
    print("G: close hand | F: open hand | SPACE: start/stop recording | S: save and quit")

    try:
        while viewer.is_running():
            with lock:
                targets = arm_targets.copy()
                desired_recording = recording_requested[0]
                should_save = save_requested[0]
                save_requested[0] = False
                close_hand = close_hand_requested[0]
                open_hand = open_hand_requested[0]
                close_hand_requested[0] = False
                open_hand_requested[0] = False

            data.ctrl[:] = initial_controls
            for actuator_id, target in zip(arm_actuator_ids, targets):
                lower, upper = model.actuator_ctrlrange[actuator_id]
                data.ctrl[actuator_id] = np.clip(target, lower, upper)

            if close_hand:
                hand_closed = True
            elif open_hand:
                hand_closed = False
            hand_targets = closed_hand_targets if hand_closed else open_hand_targets
            for actuator_id, target in zip(hand_actuator_ids, hand_targets):
                data.ctrl[actuator_id] = target

            if desired_recording != recording:
                recording = desired_recording
                if recording:
                    next_record_time = data.time
                print(f"[STATUS] {'RECORDING' if recording else 'PAUSED'}")

            mujoco.mj_step(model, data)
            _set_fixed_base(model, data, base_pose)

            if recording and data.time >= next_record_time:
                frames.append(data.qpos.copy())
                next_record_time = data.time + 1.0 / record_hz

            viewer.sync()
            if should_save:
                break
            time.sleep(model.opt.timestep)
    finally:
        viewer.close()

    if not should_save:
        return None
    if not frames:
        print("[WARNING] Nothing to save.")
        return None

    SKILLS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    saved_path = SKILLS_DIRECTORY / f"{filename}-demo.npz"
    np.savez_compressed(
        saved_path,
        qpos_seq=np.asarray(frames, dtype=np.float64),
        record_hz=np.asarray(record_hz, dtype=np.float64),
    )
    register_learned_skill(task_name, saved_path)
    print(f"[SAVED] Learned '{task_name}' ({len(frames)} frames) at {saved_path}")
    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--xml", default=str(DEFAULT_SCENE))
    parser.add_argument("--record-hz", type=float, default=30.0)
    args = parser.parse_args()
    record_task(args.task, args.xml, args.record_hz)


if __name__ == "__main__":
    main()
