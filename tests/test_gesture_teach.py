import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from teleop import gesture_teach
from teleop import executor, teacher


def hand_with_extended_fingers(indices):
    landmarks = [
        SimpleNamespace(x=0.5, y=0.5)
        for _ in range(21)
    ]
    landmarks[4] = SimpleNamespace(x=0.2, y=0.5)
    landmarks[5] = SimpleNamespace(x=0.22, y=0.5)
    for tip, pip in zip((8, 12, 16, 20), (6, 10, 14, 18)):
        landmarks[tip].y = 0.2 if tip in indices else 0.8
        landmarks[pip].y = 0.5
    return SimpleNamespace(landmark=landmarks)


class GestureTeachTests(unittest.TestCase):
    def test_executor_refuses_unknown_task_with_teaching_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(teacher, "SKILLS_FILE", Path(directory) / "skills.json"):
                output = StringIO()
                with redirect_stdout(output):
                    executor.execute_task("stack_cubes")

        self.assertIn("no prior knowledge", output.getvalue())
        self.assertIn("Please navigate me through it or teach it first", output.getvalue())
        self.assertIn(
            "python -m teleop.gesture_teach --task stack_cubes --keyboard",
            output.getvalue(),
        )

    def test_executor_loads_full_scene_trajectory(self):
        model = mujoco.MjModel.from_xml_path("sim/nero_arena.xml")
        trajectory = np.zeros((2, model.nq), dtype=np.float64)
        cube_joint = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "cube_red_free",
        )
        cube_address = model.jnt_qposadr[cube_joint]
        trajectory[:, cube_address] = (-0.1, 0.1)

        with tempfile.TemporaryDirectory() as directory:
            skill_file = f"{directory}/move_cup.npz"
            registry_file = Path(directory) / "skills.json"
            np.savez(skill_file, qpos_seq=trajectory)
            with patch.object(teacher, "SKILLS_FILE", registry_file):
                teacher.save_skill("move_cup", skill_file, len(trajectory))
                loaded = executor.load_trajectory("move_cup", model)

        np.testing.assert_array_equal(loaded, trajectory)
        self.assertEqual(loaded.shape, (2, model.nq))

    def test_keyboard_controls_move_pick_place_and_save(self):
        target = np.array([0.0, 0.4, gesture_teach.UPPER_LAYER_HEIGHT])

        self.assertEqual(
            gesture_teach.apply_keyboard_command("Right", target),
            "MOVE",
        )
        self.assertAlmostEqual(target[0], 0.05)
        self.assertEqual(
            gesture_teach.apply_keyboard_command("Up", target),
            "MOVE",
        )
        self.assertAlmostEqual(target[1], 0.45)
        self.assertEqual(
            gesture_teach.apply_keyboard_command("Next", target),
            "MOVE",
        )
        self.assertAlmostEqual(
            target[2],
            gesture_teach.UPPER_LAYER_HEIGHT - 0.05,
        )
        self.assertEqual(
            gesture_teach.apply_keyboard_command("w", target),
            "MOVE",
        )
        self.assertAlmostEqual(target[2], gesture_teach.UPPER_LAYER_HEIGHT)
        self.assertEqual(
            gesture_teach.apply_keyboard_command("s", target),
            "MOVE",
        )
        self.assertAlmostEqual(
            target[2],
            gesture_teach.UPPER_LAYER_HEIGHT - 0.05,
        )
        self.assertEqual(gesture_teach.apply_keyboard_command("p", target), "PINCH")
        self.assertEqual(
            gesture_teach.apply_keyboard_command("o", target),
            "OPEN_PALM",
        )
        self.assertEqual(
            gesture_teach.apply_keyboard_command("Return", target),
            "SAVE",
        )

    def test_shift_arrows_nudge_held_object_not_arm_target(self):
        target = np.array([0.0, 0.4, gesture_teach.UPPER_LAYER_HEIGHT])
        held_offset = np.array([0.0, 0.0, -0.04])
        original_target = target.copy()

        self.assertEqual(
            gesture_teach.apply_keyboard_command(
                "Right",
                target,
                shift=True,
                held_offset=held_offset,
            ),
            "MOVE_OBJECT",
        )
        self.assertAlmostEqual(held_offset[0], 0.05)
        np.testing.assert_array_equal(target, original_target)
        self.assertEqual(
            gesture_teach.apply_keyboard_command(
                "Up",
                target,
                shift=True,
                held_offset=held_offset,
            ),
            "MOVE_OBJECT",
        )
        self.assertAlmostEqual(held_offset[1], 0.05)
        self.assertEqual(
            gesture_teach.apply_keyboard_command(
                "Left",
                target,
                shift=True,
                held_offset=None,
            ),
            "NO_OBJECT_HELD",
        )

    def test_gesture_names_cover_move_lower_raise_pick_place_and_save(self):
        self.assertEqual(
            gesture_teach.get_gesture(hand_with_extended_fingers({8})),
            "POINT",
        )
        self.assertEqual(
            gesture_teach.get_gesture(hand_with_extended_fingers({8, 12})),
            "LOWER",
        )
        self.assertEqual(
            gesture_teach.get_gesture(hand_with_extended_fingers({8, 12, 16})),
            "RAISE",
        )
        self.assertEqual(
            gesture_teach.get_gesture(hand_with_extended_fingers({8, 12, 16, 20})),
            "OPEN_PALM",
        )
        pinch = hand_with_extended_fingers(set())
        pinch.landmark[4].x = pinch.landmark[8].x
        pinch.landmark[4].y = pinch.landmark[8].y
        self.assertEqual(gesture_teach.get_gesture(pinch), "PINCH")
        thumb_up = hand_with_extended_fingers(set())
        thumb_up.landmark[4].x = 0.05
        self.assertEqual(gesture_teach.get_gesture(thumb_up), "RAISE")
        self.assertEqual(
            gesture_teach.get_gesture(hand_with_extended_fingers(set())),
            "FIST",
        )

    def test_arena_has_reachable_movable_ingredients_and_renders(self):
        model = mujoco.MjModel.from_xml_path("sim/nero_arena.xml")
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        for body_name in gesture_teach.GRASPABLE_BODIES + ("plate",):
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{body_name}_free",
            )
            self.assertGreaterEqual(body_id, 0)
            self.assertGreaterEqual(joint_id, 0)

        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stacking_box"),
            0,
        )
        self.assertTrue(np.allclose(model.jnt_range[0], [-2.96, 2.96]))

        renderer = mujoco.Renderer(model, height=600, width=800)
        try:
            renderer.update_scene(data, camera="top_cam")
            self.assertEqual(renderer.render().shape, (600, 800, 3))
        finally:
            renderer.close()

    def test_inverse_kinematics_reaches_each_task_area(self):
        model = mujoco.MjModel.from_xml_path("sim/nero_arena.xml")
        data = mujoco.MjData(model)
        end_effector_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "nero_ee",
        )
        targets = (
            (0.12, 0.53, gesture_teach.CONTACT_HEIGHT),
            (0.52, 0.12, gesture_teach.CONTACT_HEIGHT),
            (0.35, 0.74, gesture_teach.CONTACT_HEIGHT),
            (-0.43, 0.12, gesture_teach.CONTACT_HEIGHT),
            (-0.48, 0.76, gesture_teach.CONTACT_HEIGHT),
        )

        for target in targets:
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            for _ in range(250):
                gesture_teach._move_arm_toward(
                    model,
                    data,
                    end_effector_id,
                    np.asarray(target),
                )
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.xpos[end_effector_id], target, atol=0.03)


if __name__ == "__main__":
    unittest.main()
