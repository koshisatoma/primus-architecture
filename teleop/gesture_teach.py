"""Teach Nero simulated pick-and-place tasks with gestures or keyboard input."""

import argparse
import os
import time
import tkinter as tk

import cv2
import mediapipe as mp
import mujoco
import numpy as np
from PIL import Image, ImageTk

from teleop.teacher import save_skill


TASKS = {
    "stack_cubes": "Pick up both cubes and stack them inside the open box.",
    "move_cup": "Move the orange cup to the green target on the table.",
    "glass_to_plate": "Pick up the glass and place it on the plate.",
    "pick_cup": "Move the orange cup to the green target on the table.",
}
GRIPPER_JOINTS = (7, 8)
GRASPABLE_BODIES = ("cube_red", "cube_blue", "cup", "glass")
OBJECT_GRASP_OFFSET = np.array([0.0, 0.0, -0.04])
HOVER_HEIGHT = 1.08
CONTACT_HEIGHT = 0.91
UPPER_LAYER_HEIGHT = 0.99


def apply_keyboard_command(key, target_pos, *, shift=False, held_offset=None):
    """Move the arm, or nudge a held object with Shift plus a movement key."""
    movement = {
        "Left": (-0.05, 0.0, 0.0),
        "Right": (0.05, 0.0, 0.0),
        "Up": (0.0, 0.05, 0.0),
        "Down": (0.0, -0.05, 0.0),
        "Prior": (0.0, 0.0, 0.05),
        "Next": (0.0, 0.0, -0.05),
        "w": (0.0, 0.0, 0.05),
        "s": (0.0, 0.0, -0.05),
    }
    if key in movement:
        if shift:
            if held_offset is None:
                return "NO_OBJECT_HELD"
            held_offset[:] = np.clip(
                held_offset + np.asarray(movement[key]),
                [-0.5, -0.5, -0.5],
                [0.5, 0.5, 0.5],
            )
            return "MOVE_OBJECT"
        target_pos[:] = np.clip(
            target_pos + np.asarray(movement[key]),
            [-0.55, 0.05, CONTACT_HEIGHT],
            [0.55, 0.95, UPPER_LAYER_HEIGHT],
        )
        return "MOVE"
    return {
        "p": "PINCH",
        "o": "OPEN_PALM",
        "Return": "SAVE",
    }.get(key, "NEUTRAL")


def get_gesture(landmarks):
    """Classify a hand as a point, lower, pinch, open hand, fist, or neutral."""
    points = landmarks.landmark
    finger_tips = (8, 12, 16, 20)
    finger_pips = (6, 10, 14, 18)
    extended = [
        points[tip].y < points[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    ]

    thumb_index_distance = np.hypot(
        points[4].x - points[8].x,
        points[4].y - points[8].y,
    )
    if thumb_index_distance < 0.055:
        return "PINCH"
    extended_count = sum(extended)
    if extended_count == 1 and extended[0]:
        return "POINT"
    if extended_count == 2 and extended[:2] == [True, True]:
        return "LOWER"
    if extended_count == 3 and extended[:3] == [True, True, True]:
        return "RAISE"
    if extended_count >= 4:
        return "OPEN_PALM"
    thumb_extended = np.hypot(
        points[4].x - points[5].x,
        points[4].y - points[5].y,
    ) > 0.12
    if extended_count == 0 and thumb_extended:
        return "RAISE"
    if extended_count == 0:
        return "FIST"
    return "NEUTRAL"


def _move_arm_toward(model, data, end_effector_id, target_pos):
    mujoco.mj_forward(model, data)
    current_pos = data.xpos[end_effector_id].copy()
    jacobian = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacobian, None, current_pos, end_effector_id)
    joint_delta = np.linalg.pinv(jacobian[:, :7]) @ (target_pos - current_pos)
    data.qpos[:7] += np.clip(joint_delta, -0.04, 0.04)

    for joint_index in range(7):
        lower, upper = model.jnt_range[joint_index]
        data.qpos[joint_index] = np.clip(
            data.qpos[joint_index],
            lower,
            upper,
        )


def _free_joint_qpos_address(model, body_name):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"{body_name}_free",
    )
    if joint_id == -1:
        raise ValueError(f"Movable object {body_name!r} has no free joint")
    return int(model.jnt_qposadr[joint_id])


def teach_task(task_name, camera_index=0, overwrite=False, keyboard=False):
    """Record a full MuJoCo qpos trajectory for one named task."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task {task_name!r}; choose from {', '.join(TASKS)}")

    save_path = f"data/skills/{task_name}_nero.npz"
    if os.path.exists(save_path) and not overwrite:
        raise FileExistsError(
            f"{save_path} already exists. Pass --overwrite to replace this recording."
        )

    cap = None if keyboard else cv2.VideoCapture(camera_index)
    use_camera = cap is not None and cap.isOpened()
    if not use_camera:
        if cap is not None:
            cap.release()
            cap = None
        print(
            f"Webcam {camera_index} is unavailable; starting keyboard teaching instead."
        )

    hands = None
    renderer = None
    root = None
    running = True
    try:
        if use_camera:
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
        model = mujoco.MjModel.from_xml_path("sim/nero_arena.xml")
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[list(GRIPPER_JOINTS)] = 0.04
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=600, width=800)
        end_effector_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "nero_ee",
        )
        if end_effector_id == -1:
            raise ValueError("The simulation model has no nero_ee end-effector body")
        object_qpos = {
            name: _free_joint_qpos_address(model, name)
            for name in GRASPABLE_BODIES
        }
        target_pos = np.array(
            [0.0, 0.4, HOVER_HEIGHT if use_camera else UPPER_LAYER_HEIGHT]
        )
        trajectory = []
        is_recording = False
        held_body = None
        held_offset = [None]
        was_pinching = False
        gripper_open = 0.04
        gripper_closed = 0.012
        keyboard_gesture = ["NEUTRAL"]

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0.35, 1.05]
        camera.distance = 3.8
        camera.azimuth = 145
        camera.elevation = -25

        root = tk.Tk()
        root.title(f"Nero teaching: {task_name}")
        root.attributes("-fullscreen", True)
        root.configure(bg="black")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        preview_width = min(1400, screen_width, (screen_height - 120) * 4 // 3)
        preview_height = preview_width * 3 // 4
        image_label = tk.Label(root, bg="black")
        image_label.pack(expand=True)
        status_label = tk.Label(
            root,
            text=TASKS[task_name],
            font=("TkDefaultFont", 16),
            padx=8,
            pady=6,
            bg="black",
            fg="white",
        )
        status_label.pack()
        camera_controls = (
            "Index: move  |  V-sign: contact  |  3 fingers/thumb-up: upper layer  |  "
            "Pinch: pick  |  Open palm: place  |  Fist: save"
        )
        keyboard_controls = (
            "Arrows: move  |  W/S or PageUp/PageDown: raise/lower  |  "
            "P: pick  |  Shift+arrows: nudge held object  |  "
            "O: place  |  Enter: save"
        )
        controls_label = tk.Label(
            root,
            text=(
                f"Drag: orbit  |  Wheel: zoom  |  "
                f"{camera_controls if use_camera else keyboard_controls}  |  "
                "Esc: cancel"
            ),
            font=("TkDefaultFont", 12),
            padx=8,
            pady=6,
            bg="black",
            fg="#cccccc",
        )
        controls_label.pack()

        def close_window(_event=None):
            nonlocal running
            running = False
            return "break"

        mouse_position = [None, None]

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
            camera.distance = float(np.clip(camera.distance * (1.0 + direction * 0.08), 1.5, 8.0))

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

        def handle_keyboard(event):
            nonlocal is_recording
            is_shift = bool(event.state & 0x0001)
            gesture = apply_keyboard_command(
                event.keysym,
                target_pos,
                shift=is_shift,
                held_offset=held_offset[0],
            )
            if gesture == "NEUTRAL":
                gesture = apply_keyboard_command(
                    event.keysym.lower(),
                    target_pos,
                    shift=is_shift,
                    held_offset=held_offset[0],
                )
            if gesture in {"MOVE", "MOVE_OBJECT", "PINCH", "OPEN_PALM"}:
                if not is_recording:
                    is_recording = True
                    print(">>> RECORDING STARTED")
                keyboard_gesture[0] = gesture
            elif gesture == "NO_OBJECT_HELD":
                status_label.config(text="Pick up an object with P before using Shift+arrows.")
            elif gesture == "SAVE":
                keyboard_gesture[0] = gesture
            return "break"

        if not use_camera:
            root.bind_all("<KeyPress>", handle_keyboard)
        root.focus_force()

        print(f"=== TEACHING NERO: {task_name.upper()} ===")
        print(TASKS[task_name])
        if use_camera:
            print("Point with your index finger to move. Make a V-sign to reach the table.")
            print("Use three fingers or thumb-up to reach the upper layer.")
            print("Pinch thumb and index to pick; open your hand to place.")
            print("Make a fist to save and finish.")
        else:
            print("Arrow keys: move across the table. W: raise; S: lower.")
            print("PageUp/PageDown also raise/lower.")
            print("P: pinch/pick up. Shift+arrows: nudge the held object.")
            print("O: open/place. Enter: save and finish.")
        print("Press Esc to cancel without saving.")
        print("Drag the simulation image to orbit; use the mouse wheel to zoom.")

        while running:
            if use_camera:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(
                        f"Webcam {camera_index} stopped returning frames. "
                        "The current recording was not saved."
                    )
                frame = cv2.flip(frame, 1)
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = (
                    results.multi_hand_landmarks[0]
                    if results.multi_hand_landmarks
                    else None
                )
                gesture = get_gesture(landmarks) if landmarks is not None else "NEUTRAL"
            else:
                landmarks = None
                gesture = keyboard_gesture[0]
                keyboard_gesture[0] = "NEUTRAL"

            if (
                use_camera
                and gesture in {"POINT", "LOWER", "RAISE", "PINCH"}
                and not is_recording
            ):
                is_recording = True
                print(">>> RECORDING STARTED")

            if landmarks is not None:
                index_tip = landmarks.landmark[8]
                if gesture in {"POINT", "LOWER", "RAISE", "PINCH"}:
                    target_pos[0] = float(np.clip((index_tip.x - 0.5) * 1.1, -0.55, 0.55))
                    target_pos[1] = float(np.clip(1.0 - index_tip.y * 0.9, 0.05, 0.95))
                if gesture == "POINT":
                    target_pos[2] = HOVER_HEIGHT
                elif gesture == "LOWER":
                    target_pos[2] = CONTACT_HEIGHT
                elif gesture == "RAISE":
                    target_pos[2] = UPPER_LAYER_HEIGHT

            if gesture in {"FIST", "SAVE"} and is_recording:
                if not trajectory:
                    raise RuntimeError("No motion frames were captured; nothing was saved.")
                os.makedirs("data/skills", exist_ok=True)
                np.savez(save_path, qpos_seq=np.asarray(trajectory, dtype=np.float64))
                save_skill(task_name, save_path, len(trajectory))
                print(f">>> RECORDING FINISHED. Saved {len(trajectory)} steps.")
                break

            _move_arm_toward(model, data, end_effector_id, target_pos)

            if gesture == "PINCH":
                data.qpos[list(GRIPPER_JOINTS)] = gripper_closed
                if (not was_pinching or not use_camera) and held_body is None:
                    mujoco.mj_forward(model, data)
                    end_effector_pos = data.xpos[end_effector_id]
                    nearest_name = min(
                        GRASPABLE_BODIES,
                        key=lambda name: np.linalg.norm(
                            data.xpos[
                                mujoco.mj_name2id(
                                    model,
                                    mujoco.mjtObj.mjOBJ_BODY,
                                    name,
                                )
                            ]
                            - end_effector_pos
                        ),
                    )
                    nearest_id = mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        nearest_name,
                    )
                    if np.linalg.norm(data.xpos[nearest_id] - end_effector_pos) <= 0.16:
                        held_body = nearest_name
                        held_offset[0] = (
                            data.qpos[
                                object_qpos[held_body] : object_qpos[held_body] + 3
                            ]
                            - end_effector_pos
                        ).copy()
                    else:
                        status_label.config(
                            text="No object close enough to pinch. Lower the gripper and try again."
                        )
                was_pinching = True
            elif gesture == "MOVE_OBJECT":
                pass
            elif gesture == "OPEN_PALM":
                data.qpos[list(GRIPPER_JOINTS)] = gripper_open
                held_body = None
                held_offset[0] = None
                was_pinching = False
            elif gesture == "FIST":
                data.qpos[list(GRIPPER_JOINTS)] = gripper_open
                was_pinching = False
            else:
                was_pinching = False

            mujoco.mj_forward(model, data)
            if held_body is not None:
                address = object_qpos[held_body]
                data.qpos[address : address + 3] = (
                    data.xpos[end_effector_id]
                    + (
                        held_offset[0]
                        if held_offset[0] is not None
                        else OBJECT_GRASP_OFFSET
                    )
                )
                mujoco.mj_forward(model, data)

            if is_recording:
                trajectory.append(data.qpos.copy())

            renderer.update_scene(data, camera=camera)
            sim_frame = renderer.render()
            preview_image = Image.fromarray(sim_frame).resize(
                (preview_width, preview_height),
                Image.Resampling.BILINEAR,
            )
            preview = ImageTk.PhotoImage(preview_image, master=root)
            image_label.config(image=preview)
            image_label.image = preview

            if gesture != "PINCH" or held_body is not None:
                recording_status = "RECORDING" if is_recording else "READY"
                held_status = f"  |  Holding: {held_body}" if held_body else ""
                status_label.config(
                    text=(
                        f"{task_name}: {TASKS[task_name]}  |  "
                        f"{recording_status}  |  Gesture: {gesture}{held_status}"
                    )
                )
            root.update_idletasks()
            root.update()
            time.sleep(0.01)
    finally:
        if cap is not None:
            cap.release()
        if hands is not None:
            hands.close()
        if root is not None:
            root.destroy()
        if renderer is not None:
            renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="teach with keyboard controls instead of a webcam",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing recording for this task",
    )
    args = parser.parse_args()
    teach_task(
        args.task,
        camera_index=args.camera,
        overwrite=args.overwrite,
        keyboard=args.keyboard,
    )


if __name__ == "__main__":
    main()
