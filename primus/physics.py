import math

import numpy as np


class PhysicsScore:
    def __init__(self, model, data):
        self.m = model
        self.d = data
        self.falls = 0
        self.uptime = 0.0
        self.balance_margin = 0.0
        self.contact_stability = 0.0
        self.torque_headroom = 1.0
        self.score = 0.0
        self._was_fallen = self._com_height() < 0.35

    def _com_height(self):
        if self.d.subtree_com.shape[0] < 2:
            raise ValueError('The physics model has no robot subtree COM data')
        return float(self.d.subtree_com[1, 2])

    def _supporting_feet(self):
        supported = set()
        foot_bodies = {'left_ankle_roll_link', 'right_ankle_roll_link'}
        for index in range(self.d.ncon):
            contact = self.d.contact[index]
            body1 = int(self.m.geom_bodyid[contact.geom1])
            body2 = int(self.m.geom_bodyid[contact.geom2])
            if body1 == 0 and body2 != 0:
                body1, body2 = body2, body1
            if body2 == 0:
                name = self.m.body(body1).name
                if name in foot_bodies:
                    supported.add(name)
        return len(supported)

    def step(self, dt=None):
        if dt is None:
            dt = float(self.m.opt.timestep)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError(f'Physics timestep must be finite and non-negative: {dt}')

        self.uptime += dt
        com_z = self._com_height()
        fallen = com_z < 0.35
        if fallen and not self._was_fallen:
            self.falls += 1
        self._was_fallen = fallen

        contacts = int(self.d.ncon)
        supporting_feet = self._supporting_feet()
        self.balance_margin = float(np.clip(com_z / 1.0, 0.0, 1.0))
        self.contact_stability = min(1.0, supporting_feet / 2.0)
        actuator_forces = np.abs(self.d.qfrc_actuator)
        max_torque = float(actuator_forces.max()) if actuator_forces.size else 0.0
        self.torque_headroom = max(0.0, 1.0 - max_torque / 100.0)
        self.score = 100.0 * (
            0.4 * self.balance_margin
            + 0.25 * self.contact_stability
            + 0.2 * self.torque_headroom
            + 0.15 * (0.0 if fallen else 1.0)
        )

        return {
            'com_z': com_z,
            'contacts': contacts,
            'supporting_feet': supporting_feet,
            'balance': self.balance_margin,
            'contact_stab': self.contact_stability,
            'torque_head': self.torque_headroom,
            'score': self.score,
            'uptime': self.uptime,
            'falls': self.falls,
            'fallen': fallen,
        }
