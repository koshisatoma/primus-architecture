"""Replay a saved Nero gesture-teaching trajectory in the 3D simulation."""

import argparse
import time
import tkinter as tk

import mujoco
import numpy as np
from PIL import Image, ImageTk

from teleop.teacher import check_skill, load_registry


def load_trajectory(task_name, model):
    """Load and validate a task's complete generalized-position trajectory."""
    exists, message = check_skill(task_name)
    if not exists:
        raise FileNotFoundError(message)

    registry = load_registry()
    task_key = task_name.lower().strip()
    trajectory_path = registry[task_key]["file_path"]
    with np.load(trajectory_path, allow_pickle=False) as archive:
        if "qpos_seq" not in archive:
            raise ValueError(f"Skill trajectory has no qpos_seq: {trajectory_path}")
        trajectory = np.asarray(archive["qpos_seq"], dtype=np.float64)

    if trajectory.ndim != 2 or trajectory.shape[0] == 0:
        raise ValueError(f"Skill trajectory is empty or invalid: {trajectory_path}")
    if trajectory.shape[1] != model.nq:
        raise ValueError(
            f"Trajectory has {trajectory.shape[1]} qpos values per frame; "
            f"the scene requires {model.nq}"
        )
    if not np.isfinite(trajectory).all():
        raise ValueError(f"Skill trajectory contains non-finite values: {trajectory_path}")
    return trajectory


def execute_task(task_name, speed=1.0):
    """Replay the saved complete scene state, including object positions."""
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Playback speed must be a positive finite number")

    model = mujoco.MjModel.from_xml_path("sim/nero_arena.xml")
    trajectory = load_trajectory(task_name, model)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    renderer = mujoco.Renderer(model, height=600, width=800)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.35, 1.05]
    camera.distance = 3.8
    camera.azimuth = 145
    camera.elevation = -25

    root = tk.Tk()
    root.title(f"Nero playback: {task_name}")
    root.attributes("-fullscreen", True)
    root.configure(bg="black")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    preview_width = min(1400, screen_width, (screen_height - 100) * 4 // 3)
    preview_height = preview_width * 3 // 4
    image_label = tk.Label(root, bg="black")
    image_label.pack(expand=True)
    status_label = tk.Label(
        root,
        font=("TkDefaultFont", 16),
        padx=8,
        pady=6,
        bg="black",
        fg="white",
    )
    status_label.pack()
    controls_label = tk.Label(
        root,
        text="Drag: orbit  |  Wheel: zoom  |  Space: pause/resume  |  +/-: speed  |  Esc: quit",
        font=("TkDefaultFont", 12),
        padx=8,
        pady=6,
        bg="black",
        fg="#cccccc",
    )
    controls_label.pack()

    running = True
    paused = False
    playback_speed = [speed]
    mouse_position = [None, None]
    frame_index = 0
    frame_period = 0.01
    next_frame_at = time.monotonic()
    completed = False

    def close_window(_event=None):
        nonlocal running
        running = False
        return "break"

    def toggle_pause(_event=None):
        nonlocal paused
        paused = not paused
        return "break"

    def change_speed(event):
        factor = 1.2 if event.keysym in ("plus", "equal") else 1 / 1.2
        playback_speed[0] = float(np.clip(playback_speed[0] * factor, 0.1, 4.0))
        return "break"

    def reset_playback(_event=None):
        nonlocal frame_index, next_frame_at, completed, paused
        frame_index = 0
        next_frame_at = time.monotonic()
        completed = False
        paused = False
        return "break"

    def begin_orbit(event):
        mouse_position[:] = [event.x, event.y]

    def orbit(event):
        if mouse_position[0] is not None:
            camera.azimuth += (event.x - mouse_position[0]) * 0.4
            camera.elevation = float(
                np.clip(
                    camera.elevation + (event.y - mouse_position[1]) * 0.3,
                    -85,
                    85,
                )
            )
            mouse_position[:] = [event.x, event.y]

    def end_orbit(_event):
        mouse_position[:] = [None, None]

    def zoom(event):
        direction = -1 if event.delta > 0 else 1
        camera.distance = float(
            np.clip(camera.distance * (1.0 + direction * 0.08), 1.5, 8.0)
        )

    def zoom_linux_in(_event):
        camera.distance = max(1.5, camera.distance * 0.92)

    def zoom_linux_out(_event):
        camera.distance = min(8.0, camera.distance * 1.08)

    image_label.bind("<ButtonPress-1>", begin_orbit)
    image_label.bind("<B1-Motion>", orbit)
    image_label.bind("<ButtonRelease-1>", end_orbit)
    image_label.bind("<MouseWheel>", zoom)
    image_label.bind("<Button-4>", zoom_linux_in)
    image_label.bind("<Button-5>", zoom_linux_out)
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.bind("<Escape>", close_window)
    root.bind("<KeyPress-q>", close_window)
    root.bind("<space>", toggle_pause)
    root.bind("<KeyPress-plus>", change_speed)
    root.bind("<KeyPress-equal>", change_speed)
    root.bind("<KeyPress-minus>", change_speed)
    root.bind("<KeyPress-r>", reset_playback)
    tk.Button(
        root,
        text="Reset / Replay",
        command=reset_playback,
        font=("TkDefaultFont", 14),
        padx=18,
        pady=6,
    ).pack(pady=(0, 10))
    root.focus_force()

    print(f"=== EXECUTING: {task_name.upper()} ===")
    print(f"Loaded {len(trajectory)} full-scene frames.")
    print("Drag to orbit; Space pauses; +/- changes speed; Esc quits.")

    try:
        while running:
            root.update_idletasks()
            root.update()
            if not running:
                continue
            if frame_index >= len(trajectory):
                if not completed:
                    print(">>> Execution complete! Press Reset / Replay or R.")
                    status_label.config(text=f"{task_name}: complete — press Reset / Replay")
                    completed = True
                time.sleep(0.01)
                continue
            if paused:
                next_frame_at = time.monotonic()
                time.sleep(0.01)
                continue

            now = time.monotonic()
            if now < next_frame_at:
                time.sleep(min(0.005, next_frame_at - now))
                continue

            data.qpos[:] = trajectory[frame_index]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render()).resize(
                (preview_width, preview_height),
                Image.Resampling.BILINEAR,
            )
            preview = ImageTk.PhotoImage(frame, master=root)
            image_label.config(image=preview)
            image_label.image = preview

            status_label.config(
                text=(
                    f"{task_name}  |  Frame {frame_index + 1}/{len(trajectory)}"
                    f"  |  {'PAUSED' if paused else 'PLAYING'}"
                    f"  |  {playback_speed[0]:.1f}x"
                )
            )
            frame_index += 1
            next_frame_at = now + frame_period / playback_speed[0]

    finally:
        root.destroy()
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="name of the taught task")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier (default: 1.0)",
    )
    args = parser.parse_args()
    execute_task(args.task, speed=args.speed)


if __name__ == "__main__":
    main()
