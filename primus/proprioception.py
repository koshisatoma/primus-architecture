"""Read-only proprioceptive helpers for MuJoCo model/data pairs."""

import time

import mujoco
import numpy as np


class BodySchema:
    def __init__(self, model, data):
        self.m = model
        self.d = data
        self.torso = self._get_body_id('torso_link')
        if self.torso < 0:
            raise ValueError("G1 model is missing the 'torso_link' body")
        self.left_hand = self._get_body_id('left_wrist_yaw_link')
        self.right_hand = self._get_body_id('right_wrist_yaw_link')

    def _get_body_id(self, name):
        return mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, name
        )

    def get_end_effectors(self):
        return {
            side: self.d.xpos[body_id].copy()
            for side, body_id in (
                ('left', self.left_hand),
                ('right', self.right_hand),
            )
            if body_id >= 0
        }

    def get_self_joints(self):
        return self.d.qpos.copy()

    def is_moving(self, history_qpos, threshold=0.01):
        if history_qpos is None:
            return False
        previous = np.asarray(history_qpos)
        return previous.shape == self.d.qpos.shape and bool(
            np.max(np.abs(self.d.qpos - previous)) > threshold
        )


class FallDetector:
    def __init__(
        self,
        model,
        data,
        height_threshold: float = 0.4,
        upright_threshold: float = 0.5,
        confirmation_seconds: float = 1.0,
    ):
        if confirmation_seconds < 0:
            raise ValueError('confirmation_seconds must be non-negative')
        self.m = model
        self.d = data
        self.torso = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link'
        )
        if self.torso < 0:
            raise ValueError("G1 model is missing the 'torso_link' body")
        self.height_threshold = height_threshold
        self.upright_threshold = upright_threshold
        self.confirmation_seconds = confirmation_seconds
        self.state = 'standing'
        self.fall_time = None

    def update(self, now=None):
        timestamp = time.monotonic() if now is None else float(now)
        height = float(self.d.xpos[self.torso, 2])
        rotation = self.d.xmat[self.torso].reshape(3, 3)
        upright = float(rotation[2, 2])
        if height < self.height_threshold or upright < self.upright_threshold:
            if self.state == 'standing':
                self.state = 'falling'
                self.fall_time = timestamp
            elif (
                self.state == 'falling'
                and self.fall_time is not None
                and timestamp - self.fall_time >= self.confirmation_seconds
            ):
                self.state = 'fallen'
        else:
            self.state = 'standing'
            self.fall_time = None
        return self.state


class AuditoryLocalization:
    """A smoothed interaural-level heuristic; no microphone geometry is inferred."""

    def __init__(self, smoothing: float = 0.2):
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError('smoothing must be in [0, 1]')
        self.smoothing = smoothing
        self.current_yaw = 0.0
        self.current_pitch = 0.0

    def update(self, audio_channels_rms):
        channels = np.asarray(audio_channels_rms, dtype=np.float64).reshape(-1)
        if channels.size < 2:
            return self.current_yaw, self.current_pitch
        if not np.isfinite(channels).all() or np.any(channels < 0):
            raise ValueError('audio channel RMS values must be finite and non-negative')
        left, right = channels[:2]
        ild = (right - left) / (right + left + 1e-6)
        target_yaw = float(np.clip(ild * 1.5, -1.0, 1.0))
        self.current_yaw = (
            (1.0 - self.smoothing) * self.current_yaw
            + self.smoothing * target_yaw
        )
        return self.current_yaw, self.current_pitch
