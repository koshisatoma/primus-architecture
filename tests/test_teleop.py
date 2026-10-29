import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from teleop import executor, recorder, teacher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TELEOP_SCENE = PROJECT_ROOT / "sim" / "g1" / "scene_teleop.xml"


class TeleopTests(unittest.TestCase):
    def test_unknown_task_is_refused_until_a_demo_is_registered(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = Path(temporary_directory) / "skills.json"
            with patch.object(teacher, "SKILLS_FILE", registry):
                response = teacher.check_and_respond("pour water")

        self.assertEqual(
            response,
            "I haven't been taught about pour water. Please navigate me to perform it.",
        )

    def test_registration_preserves_other_skills_and_normalizes_task_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = Path(temporary_directory) / "skills.json"
            trajectory = Path(temporary_directory) / "pour-demo.npz"
            trajectory.touch()
            with patch.object(teacher, "SKILLS_FILE", registry):
                teacher.save_skills({"existing": {"kind": "action"}})
                teacher.register_learned_skill("  Pour Water  ", trajectory)
                skills = teacher.load_skills()

                self.assertIn("existing", skills)
                self.assertEqual(skills["pour water"]["status"], "learned")
                self.assertEqual(
                    teacher.check_and_respond("pour water"),
                    "Executing learned skill: pour water.",
                )
                self.assertEqual(
                    teacher.resolve_trajectory_path(
                        teacher.learned_skill("pour water")
                    ),
                    trajectory,
                )

    def test_invalid_registry_shape_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = Path(temporary_directory) / "skills.json"
            registry.write_text(
                json.dumps(["not", "an", "object"]),
                encoding="utf-8",
            )
            with patch.object(teacher, "SKILLS_FILE", registry):
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    teacher.load_skills()

    def test_g1_teleop_scene_has_arm_hand_and_sandbox_objects(self):
        model = mujoco.MjModel.from_xml_path(str(TELEOP_SCENE))
        for joint_name in recorder.RIGHT_ARM_JOINTS + recorder.RIGHT_HAND_JOINTS:
            self.assertGreaterEqual(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    joint_name,
                ),
                0,
            )
        for body_name in ("red_block", "blue_cup", "green_ball"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name),
                0,
            )

    def test_position_actuator_moves_right_arm_while_base_is_held(self):
        model = mujoco.MjModel.from_xml_path(str(TELEOP_SCENE))
        data = mujoco.MjData(model)
        keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        base_pose = data.qpos[:7].copy()
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "right_elbow_joint",
        )
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "right_elbow_joint",
        )
        initial_position = float(data.qpos[model.jnt_qposadr[joint_id]])
        data.ctrl[actuator_id] += 0.3
        for _ in range(500):
            mujoco.mj_step(model, data)
            recorder._set_fixed_base(model, data, base_pose)

        self.assertGreater(
            abs(float(data.qpos[model.jnt_qposadr[joint_id]]) - initial_position),
            0.1,
        )
        np.testing.assert_array_equal(data.qpos[:7], base_pose)

    def test_recorder_saves_g1_joint_and_hand_demonstration(self):
        model = mujoco.MjModel.from_xml_path(str(TELEOP_SCENE))
        keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        stand = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, stand, keyframe_id)

        class FakeViewer:
            def __init__(self, key_callback):
                self.key_callback = key_callback
                self.ticks = 0

            def is_running(self):
                return self.ticks < 9

            def sync(self):
                if self.ticks == 0:
                    self.key_callback(ord("2"))
                    self.key_callback(recorder.KEY_UP)
                    self.key_callback(ord("g"))
                    self.key_callback(recorder.KEY_SPACE)
                elif self.ticks == 6:
                    self.key_callback(recorder.KEY_SPACE)
                elif self.ticks == 7:
                    self.key_callback(ord("s"))
                self.ticks += 1

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            registry = temporary_path / "skills.json"
            skill_directory = temporary_path / "skills"
            with (
                patch.object(teacher, "SKILLS_FILE", registry),
                patch.object(recorder, "SKILLS_DIRECTORY", skill_directory),
                patch(
                    "teleop.recorder.mujoco.viewer.launch_passive",
                    side_effect=lambda _model, _data, key_callback: FakeViewer(
                        key_callback
                    ),
                ),
                patch("teleop.recorder.time.sleep"),
            ):
                saved_path = recorder.record_task(
                    "wave",
                    TELEOP_SCENE,
                    record_hz=500,
                )
                registered_skill = teacher.load_skills()["wave"]

            self.assertIsNotNone(saved_path)
            self.assertTrue(saved_path.is_file())
            with np.load(saved_path, allow_pickle=False) as archive:
                frames = archive["qpos_seq"]

            self.assertGreaterEqual(len(frames), 2)
            self.assertEqual(frames.shape[1], model.nq)
            np.testing.assert_allclose(
                frames[:, :7],
                np.repeat(stand.qpos[:7][None, :], len(frames), axis=0),
            )
            shoulder_joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_roll_joint"
            )
            shoulder_positions = frames[:, model.jnt_qposadr[shoulder_joint_id]]
            self.assertGreater(
                np.max(
                    np.abs(
                        shoulder_positions
                        - stand.qpos[model.jnt_qposadr[shoulder_joint_id]]
                    )
                ),
                0.002,
            )
            hand_joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                "right_hand_index_0_joint",
            )
            self.assertGreater(
                np.max(frames[:, model.jnt_qposadr[hand_joint_id]]),
                stand.qpos[model.jnt_qposadr[hand_joint_id]],
            )
            self.assertEqual(registered_skill["status"], "learned")

    def test_executor_replays_a_registered_g1_demonstration(self):
        model = mujoco.MjModel.from_xml_path(str(TELEOP_SCENE))
        first_frame = model.qpos0.copy()
        second_frame = first_frame.copy()
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "right_elbow_joint",
        )
        second_frame[model.jnt_qposadr[joint_id]] = 0.4
        expected_frames = np.stack((first_frame, second_frame))

        class FakeViewer:
            def __init__(self, data):
                self.data = data
                self.frames = []
                self.closed = False

            def is_running(self):
                return len(self.frames) < len(expected_frames)

            def sync(self):
                self.frames.append(self.data.qpos.copy())

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            registry = temporary_path / "skills.json"
            trajectory = temporary_path / "demo.npz"
            np.savez(
                trajectory,
                qpos_seq=expected_frames,
                record_hz=np.asarray(30.0),
            )
            with patch.object(teacher, "SKILLS_FILE", registry):
                teacher.register_learned_skill("test task", trajectory)
                viewer = FakeViewer(None)

                def create_viewer(_model, data):
                    viewer.data = data
                    return viewer

                with (
                    patch(
                        "teleop.executor.mujoco.viewer.launch_passive",
                        side_effect=create_viewer,
                    ),
                    patch("teleop.executor.time.sleep"),
                ):
                    executor.execute_task("test task")

        self.assertTrue(viewer.closed)
        self.assertEqual(len(viewer.frames), len(expected_frames))
        for actual, expected in zip(viewer.frames, expected_frames):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
