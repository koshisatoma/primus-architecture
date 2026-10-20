from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / 'sim' / 'g1' / 'scene_mjx.xml'


class G1Env:
    """MJX G1 environment using the Menagerie position-servo actuators."""

    VALID_TASKS = {
        'stand',
        'walk',
        'walk_to',
        'sit',
        'reach',
        'grasp',
        'lift',
        'place',
    }

    def __init__(self, task='stand', xml=XML, target=None, obj_body=None):
        if task not in self.VALID_TASKS:
            raise ValueError(f'Unsupported G1 task: {task}')

        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.mx = mjx.put_model(self.m)
        self.task = task
        self.target = target
        self.obj = obj_body
        self.nq, self.nv, self.nu = self.m.nq, self.m.nv, self.m.nu
        self.obs_dim = self.nq + self.nv
        if self.nq - 7 != self.nu:
            raise ValueError(
                'G1 model must have a 7-DoF free base followed by one '
                'position actuator per joint'
            )

        self.qpos0 = jnp.asarray(self.m.qpos0)
        self.qdef = jnp.asarray(self.m.qpos0[7:7 + self.nu])
        self.ctrlrange = jnp.asarray(self.m.actuator_ctrlrange)
        self.ctrllimited = jnp.asarray(self.m.actuator_ctrllimited)
        self.torso = mujoco.mj_name2id(
            self.m,
            mujoco.mjtObj.mjOBJ_BODY,
            'torso_link',
        )
        if self.torso < 0:
            raise ValueError("G1 model is missing the 'torso_link' body")

        wrist = mujoco.mj_name2id(
            self.m,
            mujoco.mjtObj.mjOBJ_BODY,
            'left_wrist_yaw_link',
        )
        self.ee = wrist if wrist >= 0 else self.torso + 4

    def reset(self, key):
        data = mjx.make_data(self.mx)
        joint_noise = 0.01 * jax.random.normal(key, (self.nu,))
        qpos = self.qpos0.at[7:].add(joint_noise)
        data = data.replace(qpos=qpos, qvel=jnp.zeros(self.nv))
        return mjx.forward(self.mx, data)

    def step(self, data, action):
        ctrl = self.qdef + 0.5 * jnp.clip(action, -1.0, 1.0)
        bounded_ctrl = jnp.clip(ctrl, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        ctrl = jnp.where(self.ctrllimited, bounded_ctrl, ctrl)
        return mjx.step(self.mx, data.replace(ctrl=ctrl))

    def obs(self, data):
        return jnp.concatenate([data.qpos, data.qvel])

    def phi(self, data):
        rotation = data.xmat[self.torso].reshape(3, 3)
        return jnp.concatenate(
            [
                jnp.array(
                    [
                        data.xpos[self.torso, 2],
                        rotation[2, 2],
                    ]
                ),
                data.subtree_linvel[1],
            ]
        )

    def fallen(self, data):
        rotation = data.xmat[self.torso].reshape(3, 3)
        return (data.xpos[self.torso, 2] < 0.35) | (rotation[2, 2] < 0.5)

    def reward(self, data):
        rotation = data.xmat[self.torso].reshape(3, 3)
        up = rotation[2, 2]
        height = data.xpos[self.torso, 2]
        velocity = data.subtree_linvel[1]
        ee = data.xpos[self.ee]
        control_penalty = 0.001 * jnp.sum(jnp.square(data.ctrl))
        target = jnp.asarray(self.target if self.target is not None else [0.0, 0.0, 0.8])

        if self.task == 'stand':
            return 4.0 * height + 3.0 * up - control_penalty
        if self.task == 'walk' or self.task == 'walk_to':
            return (
                3.0 * velocity[0]
                + 2.0 * up
                - 2.0 * jnp.linalg.norm(data.xpos[1, :2] - target[:2])
                - control_penalty
            )
        if self.task == 'sit':
            return (
                -4.0 * jnp.abs(data.subtree_com[1, 2] - 0.45)
                + 1.5 * up
                - control_penalty
            )
        if self.task == 'reach':
            return -3.0 * jnp.linalg.norm(ee - target) + up - control_penalty
        if self.task == 'grasp':
            return (
                -3.0 * jnp.linalg.norm(ee - target)
                + 1.0 * (jnp.linalg.norm(ee - target) < 0.05)
                - control_penalty
            )
        if self.task == 'lift':
            obj_z = data.xpos[self.obj, 2] if self.obj else ee[2]
            return 4.0 * obj_z * (jnp.linalg.norm(ee - target) < 0.1) + up - control_penalty
        if self.task == 'place':
            return -3.0 * jnp.linalg.norm(data.xpos[self.obj] - target) + up - control_penalty
        return up - control_penalty
