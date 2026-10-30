#!/usr/bin/env python3
"""Launch Nero teleoperation or display its live evaluation metrics.

Usage:
    python teleop_metrics.py teach --task move_cup --keyboard
    python teleop_metrics.py replay --task move_cup
    python teleop_metrics.py monitor

Replay outcomes are operator-labeled in the monitor (S/F). Replay accuracy is
the percentage of rendered qpos frames matching a saved demonstration frame.
"""

import argparse
import fcntl
import json
import os
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
METRICS_PATH = DATA_DIR / "teleop_metrics.json"
LOCK_PATH = DATA_DIR / "teleop_metrics.lock"

DEFAULT_METRICS = {
    "mode": "idle",
    "task": None,
    "fps": 0.0,
    "fps_updated_at": None,
    "one_shot_attempts": 0,
    "one_shot_successes": 0,
    "one_shot_pending": 0,
    "replay_frames": 0,
    "replay_matches": 0,
    "zero_prior_attempts": 0,
    "zero_prior_refusals": 0,
    "teaching_saves": 0,
    "last_taught_frames": 0,
}


def _read_metrics_unlocked():
    if not METRICS_PATH.exists():
        return DEFAULT_METRICS.copy()

    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)

    if not isinstance(metrics, dict):
        raise ValueError(f"{METRICS_PATH} must contain a JSON object")

    return {**DEFAULT_METRICS, **metrics}


def _write_metrics_unlocked(metrics):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=DATA_DIR,
            prefix=".teleop-metrics-",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(metrics, file, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, METRICS_PATH)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def update_metrics(**values):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        metrics = _read_metrics_unlocked()
        metrics.update(values)
        metrics["updated_at"] = time.time()
        _write_metrics_unlocked(metrics)


def adjust_metrics(**increments):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        metrics = _read_metrics_unlocked()

        for key, amount in increments.items():
            metrics[key] = int(metrics.get(key, 0)) + amount

        metrics["updated_at"] = time.time()
        _write_metrics_unlocked(metrics)


def read_metrics():
    if not METRICS_PATH.exists():
        return DEFAULT_METRICS.copy()

    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)

    if not isinstance(metrics, dict):
        raise ValueError(f"{METRICS_PATH} must contain a JSON object")

    return {**DEFAULT_METRICS, **metrics}


def percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else 0.0


def instrument_fps(module):
    """Measure rendered Tk/PIL image updates per second."""
    original_photo_image = module.ImageTk.PhotoImage
    frame_count = 0
    window_started = time.monotonic()

    def measured_photo_image(*args, **kwargs):
        nonlocal frame_count, window_started

        photo = original_photo_image(*args, **kwargs)
        frame_count += 1

        now = time.monotonic()
        elapsed = now - window_started
        if elapsed >= 1.0:
            update_metrics(
                fps=frame_count / elapsed,
                fps_updated_at=time.time(),
            )
            frame_count = 0
            window_started = now

        return photo

    module.ImageTk.PhotoImage = measured_photo_image
    return original_photo_image


def run_teaching(args):
    from teleop import gesture_teach

    original_photo_image = instrument_fps(gesture_teach)
    original_save_skill = gesture_teach.save_skill

    def tracked_save_skill(task_name, trajectory_file, steps):
        result = original_save_skill(task_name, trajectory_file, steps)
        adjust_metrics(teaching_saves=1)
        update_metrics(last_taught_frames=steps)
        return result

    gesture_teach.save_skill = tracked_save_skill
    update_metrics(mode="teaching", task=args.task)

    try:
        gesture_teach.teach_task(
            args.task,
            camera_index=args.camera,
            overwrite=args.overwrite,
            keyboard=args.keyboard,
        )
    finally:
        gesture_teach.ImageTk.PhotoImage = original_photo_image
        gesture_teach.save_skill = original_save_skill
        update_metrics(mode="idle", task=None)

        metrics = read_metrics()
        print("\n" + "=" * 40)
        print("       TEACHING SESSION METRICS")
        print("=" * 40)
        print(f"Total Saved Skills Logs : {metrics['teaching_saves']}")
        print(f"Last Taught Frames      : {metrics['last_taught_frames']}")
        print(f"Session FPS             : {metrics['fps']:.1f}")
        print("=" * 40 + "\n")


def run_replay(args):
    import numpy as np
    from teleop import executor

    original_photo_image = instrument_fps(executor)
    original_check_skill = executor.check_skill
    original_load_trajectory = executor.load_trajectory
    original_renderer = executor.mujoco.Renderer

    attempt = {"task": args.task.casefold().strip(), "checked": False}
    expected_trajectory = {"value": None}
    pending_frames = {"count": 0, "matches": 0}

    def tracked_check_skill(task_name):
        result = original_check_skill(task_name)
        normalized = task_name.casefold().strip()

        if normalized == attempt["task"] and not attempt["checked"]:
            attempt["checked"] = True
            exists, message = result

            if exists:
                adjust_metrics(one_shot_pending=1)
            elif "no prior knowledge" in message.casefold():
                adjust_metrics(
                    zero_prior_attempts=1,
                    zero_prior_refusals=1,
                )

        return result

    def tracked_load_trajectory(task_name, model):
        trajectory = original_load_trajectory(task_name, model)
        expected_trajectory["value"] = np.asarray(trajectory)
        return trajectory

    class MeasuredRenderer:
        def __init__(self, renderer):
            self._renderer = renderer

        def __getattr__(self, name):
            return getattr(self._renderer, name)

        def update_scene(self, data, *scene_args, **scene_kwargs):
            trajectory = expected_trajectory["value"]

            if trajectory is not None:
                qpos = np.asarray(data.qpos)

                matches = bool(
                    trajectory.ndim >= 2
                    and trajectory.shape[1:] == qpos.shape
                    and np.any(
                        np.max(np.abs(trajectory - qpos), axis=1) <= 1e-8
                    )
                )

                pending_frames["count"] += 1
                pending_frames["matches"] += int(matches)

                if pending_frames["count"] >= 10:
                    adjust_metrics(
                        replay_frames=pending_frames["count"],
                        replay_matches=pending_frames["matches"],
                    )
                    pending_frames["count"] = 0
                    pending_frames["matches"] = 0

            return self._renderer.update_scene(
                data, *scene_args, **scene_kwargs
            )

        def close(self):
            return self._renderer.close()

    def measured_renderer(*renderer_args, **renderer_kwargs):
        renderer = original_renderer(*renderer_args, **renderer_kwargs)
        return MeasuredRenderer(renderer)

    executor.check_skill = tracked_check_skill
    executor.load_trajectory = tracked_load_trajectory
    executor.mujoco.Renderer = measured_renderer
    update_metrics(mode="replay", task=args.task)

    try:
        executor.execute_task(args.task, speed=args.speed)
    finally:
        if pending_frames["count"]:
            adjust_metrics(
                replay_frames=pending_frames["count"],
                replay_matches=pending_frames["matches"],
            )

        executor.ImageTk.PhotoImage = original_photo_image
        executor.check_skill = original_check_skill
        executor.load_trajectory = original_load_trajectory
        executor.mujoco.Renderer = original_renderer
        update_metrics(mode="idle", task=None)

        metrics = read_metrics()
        frames = metrics["replay_frames"]
        matches = metrics["replay_matches"]
        accuracy = percent(matches, frames)

        print("\n" + "=" * 40)
        print("        REPLAY SESSION METRICS")
        print("=" * 40)
        print(f"Total Tracked Frames   : {frames}")
        print(f"Matching Qpos Frames   : {matches}")
        print(f"Trajectory Accuracy    : {accuracy:.2f}%")
        print(f"One-Shot Pending Logs  : {metrics['one_shot_pending']}")
        print("=" * 40 + "\n")


def mark_outcome(success):
    metrics = read_metrics()

    if metrics["one_shot_pending"] <= 0:
        return False

    adjust_metrics(
        one_shot_attempts=1,
        one_shot_successes=int(success),
        one_shot_pending=-1,
    )
    return True


class TeleopLiveGraphApp:
    """Tkinter dashboard for live telemetry and metric history."""

    def __init__(self, root):
        self.root = root
        self.root.title("Nero Teleop Live Monitor Dashboard")

        screen_width = root.winfo_screenwidth()
        x_position = max(0, screen_width - 520)
        self.root.geometry(f"500x700+{x_position}+50")
        self.root.configure(bg="#1e1e1e")

        self.max_history = 50
        self.history = {
            "fps": [0.0],
            "one_shot": [0.0],
            "replay_acc": [0.0],
            "refusal_rate": [0.0],
        }
        self.peaks = {
            key: {"min": float("inf"), "max": float("-inf")}
            for key in self.history
        }

        self.text_frame = tk.Frame(
            root, bg="#1e1e1e", padx=15, pady=10
        )
        self.text_frame.pack(fill=tk.X)

        self.lbl_status = self.make_label(
            "Status: Initializing...", "#ffcc00", bold=True
        )
        self.lbl_fps = self.make_label("Control FPS: 0.0", "#00ffcc")
        self.lbl_oneshot = self.make_label(
            "One-Shot Success: 0.0%", "#ff66cc"
        )
        self.lbl_replay = self.make_label(
            "Replay Accuracy: 0.0%", "#66ccff"
        )
        self.lbl_refusal = self.make_label(
            "Zero-Prior Refusal: 0.0%", "#ff9966"
        )
        self.lbl_notice = self.make_label("", "#ffffff", italic=True)

        button_frame = tk.Frame(root, bg="#1e1e1e", pady=5)
        button_frame.pack(fill=tk.X)

        tk.Button(
            button_frame,
            text="SUCCESS (S)",
            font=("Arial", 10, "bold"),
            bg="#2eb85c",
            fg="white",
            command=lambda: self.inject_outcome(True),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        tk.Button(
            button_frame,
            text="FAIL (F)",
            font=("Arial", 10, "bold"),
            bg="#e55353",
            fg="white",
            command=lambda: self.inject_outcome(False),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        tk.Label(
            root,
            text="LIVE LOG TREND CHART (LAST 50 SAMPLES)",
            font=("Courier", 9, "bold"),
            fg="#888888",
            bg="#1e1e1e",
        ).pack(pady=(10, 0))

        self.canvas = tk.Canvas(
            root,
            width=460,
            height=320,
            bg="#2a2a2a",
            highlightthickness=0,
        )
        self.canvas.pack(pady=10, padx=15)

        root.bind("s", lambda _event: self.inject_outcome(True))
        root.bind("f", lambda _event: self.inject_outcome(False))
        root.bind("q", lambda _event: root.destroy())
        root.bind("<Escape>", lambda _event: root.destroy())

        self.update_loop()

    def make_label(self, text, color, bold=False, italic=False):
        label = tk.Label(
            self.text_frame,
            text=text,
            font=(
                "Courier",
                11 if not italic else 10,
                "bold" if bold else ("italic" if italic else "normal"),
            ),
            fg=color,
            bg="#1e1e1e",
            anchor="w",
        )
        label.pack(fill=tk.X, pady=2)
        return label

    def inject_outcome(self, success):
        if mark_outcome(success):
            outcome = "SUCCESS" if success else "FAILED"
            color = "#2eb85c" if success else "#e55353"
            self.lbl_notice.config(
                text=f"Logged: Replay marked {outcome}.",
                fg=color,
            )
        else:
            self.lbl_notice.config(
                text="Warning: No unscored pending replay logs found.",
                fg="#ffcc00",
            )

    def track_peaks(self, key, value):
        self.peaks[key]["min"] = min(self.peaks[key]["min"], value)
        self.peaks[key]["max"] = max(self.peaks[key]["max"], value)

    def update_loop(self):
        try:
            metrics = read_metrics()

            fps_value = metrics.get("fps") or 0.0
            fps_updated_at = metrics.get("fps_updated_at")
            if fps_updated_at and time.time() - fps_updated_at > 3.0:
                fps_value = 0.0

            one_shot_value = percent(
                metrics["one_shot_successes"],
                metrics["one_shot_attempts"],
            )
            replay_value = percent(
                metrics["replay_matches"],
                metrics["replay_frames"],
            )
            refusal_value = percent(
                metrics["zero_prior_refusals"],
                metrics["zero_prior_attempts"],
            )

            values = {
                "fps": fps_value,
                "one_shot": one_shot_value,
                "replay_acc": replay_value,
                "refusal_rate": refusal_value,
            }

            for key, value in values.items():
                self.history[key].append(value)
                if len(self.history[key]) > self.max_history:
                    self.history[key].pop(0)
                self.track_peaks(key, value)

            task = metrics.get("task")
            task_text = f" ({task})" if task else ""
            self.lbl_status.config(
                text=f"Status: {metrics['mode'].upper()}{task_text}"
            )

            self.format_label(
                self.lbl_fps, "Control FPS", fps_value, "fps", False
            )
            self.format_label(
                self.lbl_oneshot,
                "One-Shot Success",
                one_shot_value,
                "one_shot",
            )
            self.format_label(
                self.lbl_replay,
                "Replay Accuracy",
                replay_value,
                "replay_acc",
            )
            self.format_label(
                self.lbl_refusal,
                "Zero-Prior Refusal",
                refusal_value,
                "refusal_rate",
            )

            self.draw_chart()

        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.lbl_notice.config(text=f"Metrics read error: {error}")

        self.root.after(200, self.update_loop)

    def format_label(self, label, title, value, key, is_percent=True):
        peaks = self.peaks[key]
        suffix = "%" if is_percent else ""
        minimum = 0.0 if peaks["min"] == float("inf") else peaks["min"]
        maximum = 0.0 if peaks["max"] == float("-inf") else peaks["max"]

        label.config(
            text=(
                f"{title}: {value:.1f}{suffix} "
                f"[Range: {minimum:.1f} to {maximum:.1f}{suffix}]"
            )
        )

    def draw_chart(self):
        self.canvas.delete("all")

        width, height = 460, 320
        margin_x, margin_y = 40, 20
        graph_width = width - margin_x - 15
        graph_height = height - margin_y * 2

        self.canvas.create_rectangle(
            margin_x,
            margin_y,
            width - 15,
            height - margin_y,
            outline="#444444",
        )

        for index in range(5):
            y_position = margin_y + graph_height * index / 4
            self.canvas.create_line(
                margin_x,
                y_position,
                width - 15,
                y_position,
                fill="#3a3a3a",
                dash=(2, 2),
            )
            self.canvas.create_text(
                margin_x - 8,
                y_position,
                text=str(100 - index * 25),
                fill="#888888",
                font=("Arial", 7),
                anchor="e",
            )

        series = [
            ("fps", "#00ffcc", 60.0),
            ("one_shot", "#ff66cc", 100.0),
            ("replay_acc", "#66ccff", 100.0),
            ("refusal_rate", "#ff9966", 100.0),
        ]

        for key, color, max_scale in series:
            data = self.history[key]
            if len(data) < 2:
                continue

            points = []
            for index, value in enumerate(data):
                x = margin_x + index * (
                    graph_width / (self.max_history - 1)
                )
                ratio = min(1.0, max(0.0, value / max_scale))
                y = margin_y + graph_height * (1.0 - ratio)
                points.extend((x, y))

            self.canvas.create_line(
                points,
                fill=color,
                width=2,
                joinstyle=tk.ROUND,
            )


def monitor():
    root = tk.Tk()
    TeleopLiveGraphApp(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Nero teleoperation metrics and live dashboard"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    teach_parser = commands.add_parser(
        "teach", help="launch the teaching demo with telemetry"
    )
    teach_parser.add_argument("--task", required=True)
    teach_parser.add_argument("--camera", type=int, default=0)
    teach_parser.add_argument("--overwrite", action="store_true")
    teach_parser.add_argument("--keyboard", action="store_true")

    replay_parser = commands.add_parser(
        "replay", help="launch a saved trajectory replay with telemetry"
    )
    replay_parser.add_argument("--task", required=True)
    replay_parser.add_argument("--speed", type=float, default=1.0)

    commands.add_parser("monitor", help="show the live metrics dashboard")

    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    if args.command == "teach":
        run_teaching(args)
    elif args.command == "replay":
        run_replay(args)
    else:
        monitor()


if __name__ == "__main__":
    main()
