"""Replay a previously demonstrated task in the local MuJoCo scene."""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from teleop.teacher import (
    PROJECT_ROOT,
    check_and_respond,
    learned_skill,
    resolve_trajectory_path,
)


DEFAULT_SCENE = PROJECT_ROOT / "sim" / "g1" / "scene_teleop.xml"


def _model_path(path: str | Path) -> Path:
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path


def execute_task(
    task_name: str,
    xml_path: str | Path = DEFAULT_SCENE,
) -> None:
    response = check_and_respond(task_name)
    print(f"PRIMUS: {response}")
    skill = learned_skill(task_name)
    if skill is None:
        print(
            "Action required: run "
            f"'python -m teleop.recorder --task \"{task_name}\"' to teach it."
        )
        return

    trajectory_path = resolve_trajectory_path(skill)
    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Learned trajectory is missing: {trajectory_path}. "
            "Record the skill again to restore it."
        )

    with np.load(trajectory_path, allow_pickle=False) as archive:
        if "qpos_seq" not in archive:
            raise ValueError(f"Trajectory has no qpos_seq array: {trajectory_path}")
        trajectory = np.asarray(archive["qpos_seq"], dtype=np.float64)
        record_hz = (
            float(np.asarray(archive["record_hz"]).item())
            if "record_hz" in archive
            else 30.0
        )

    if trajectory.ndim != 2 or trajectory.shape[0] == 0:
        raise ValueError(f"Trajectory must contain at least one frame: {trajectory_path}")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"Trajectory contains non-finite joint positions: {trajectory_path}")
    if not np.isfinite(record_hz) or record_hz <= 0:
        raise ValueError(f"Trajectory has an invalid recording frequency: {trajectory_path}")

    model = mujoco.MjModel.from_xml_path(str(_model_path(xml_path)))
    if trajectory.shape[1] != model.nq:
        raise ValueError(
            f"Trajectory has {trajectory.shape[1]} position values per frame, "
            f"but the selected scene expects {model.nq}"
        )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)
    frame_period = 1.0 / record_hz
    print(f"[EXECUTOR] Replaying {len(trajectory)} frames from {trajectory_path}")

    try:
        for frame in trajectory:
            if not viewer.is_running():
                break
            data.qpos[:] = frame
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(frame_period)

        print("[EXECUTOR] Task complete.")
        hold_until = time.monotonic() + 2.0
        while viewer.is_running() and time.monotonic() < hold_until:
            viewer.sync()
            time.sleep(min(frame_period, max(0.0, hold_until - time.monotonic())))
    finally:
        viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--xml", default=str(DEFAULT_SCENE))
    args = parser.parse_args()
    execute_task(args.task, args.xml)


if __name__ == "__main__":
    main()
