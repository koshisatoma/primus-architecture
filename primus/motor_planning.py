"""Advisory navigation geometry; does not command a robot or recover falls."""

import numpy as np

from primus.proprioception import FallDetector
from primus.spatial_memory import SpatialMemory


class ClutteredLocomotion:
    """Compute a bounded local velocity suggestion from a MuJoCo snapshot."""

    def __init__(self, spatial_memory: SpatialMemory, model, data):
        import mujoco

        self.memory = spatial_memory
        self.m = model
        self.d = data
        self.torso = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link'
        )
        if self.torso < 0:
            raise ValueError("G1 model is missing the 'torso_link' body")

    def compute_command(self, goal_pos: np.ndarray) -> np.ndarray:
        goal = np.asarray(goal_pos, dtype=np.float64)
        if goal.shape not in {(2,), (3,)} or not np.isfinite(goal).all():
            raise ValueError('goal_pos must be a finite 2D or 3D vector')
        robot_pos = np.asarray(self.d.xpos[self.torso, :2], dtype=np.float64)
        rotation = self.d.xmat[self.torso].reshape(3, 3)
        robot_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        diff = goal[:2] - robot_pos
        distance = float(np.linalg.norm(diff))
        if distance < 0.1:
            return np.zeros(3, dtype=np.float64)

        force = diff / distance
        for obstacle in self.memory.get_obstacles():
            obstacle_diff = robot_pos - obstacle[:2]
            obstacle_distance = float(np.linalg.norm(obstacle_diff))
            if 0.05 < obstacle_distance < 1.0:
                magnitude = 0.5 * (1.0 / obstacle_distance - 1.0)
                magnitude /= obstacle_distance**2
                force += magnitude * obstacle_diff / obstacle_distance
        norm = float(np.linalg.norm(force))
        if norm > 1e-9:
            force /= norm
        cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
        local_x = force[0] * cos_yaw + force[1] * sin_yaw
        local_y = -force[0] * sin_yaw + force[1] * cos_yaw
        target_yaw = float(np.arctan2(diff[1], diff[0]))
        yaw_error = (target_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi
        return np.asarray(
            [
                np.clip(local_x * 0.5, -0.5, 0.5),
                np.clip(local_y * 0.2, -0.2, 0.2),
                np.clip(yaw_error * 2.0, -1.0, 1.0),
            ],
            dtype=np.float64,
        )


class FallRecoveryStateMachine:
    """Detect recovery-needed state without fabricating a recovery action."""

    def __init__(self, fall_detector: FallDetector):
        self.detector = fall_detector
        self.recovering = False

    def update(self):
        state = self.detector.update()
        self.recovering = state in {'falling', 'fallen'}
        return state

    def get_override(self, proposed_action):
        state = self.update()
        if state == 'standing':
            return proposed_action
        raise RuntimeError(
            'Fall detected; no validated get-up policy is installed. '
            'Motor commands are not overridden with a placeholder action.'
        )
