import mujoco
import numpy as np


class Rigid:
    def __init__(self, model, data):
        self.m, self.d = model, data
        self.m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        self.robot_root = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis'
        )
        self.torso = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link'
        )
        if self.robot_root < 0:
            raise ValueError("MuJoCo model is missing the G1 'pelvis' body")
        if self.torso < 0:
            raise ValueError("MuJoCo model is missing the G1 'torso_link' body")

        self._robot_bodies = {self.robot_root}
        for body_id in range(1, model.nbody):
            parent_id = int(model.body_parentid[body_id])
            while parent_id and parent_id != self.robot_root:
                parent_id = int(model.body_parentid[parent_id])
            if parent_id == self.robot_root:
                self._robot_bodies.add(body_id)

    def snap(self):
        m, d = self.m, self.d
        z_axis = d.xmat[self.torso].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))

        normal_force = 0.0
        tangential_force = 0.0
        zmp_x = 0.0
        zmp_y = 0.0
        vertical_load = 0.0
        support_contacts = 0
        contact_force = np.zeros(6)
        for index in range(d.ncon):
            contact = d.contact[index]
            body1 = int(m.geom_bodyid[contact.geom1])
            body2 = int(m.geom_bodyid[contact.geom2])
            if not (
                (body1 == 0 and body2 in self._robot_bodies)
                or (body2 == 0 and body1 in self._robot_bodies)
            ):
                continue

            mujoco.mj_contactForce(m, d, index, contact_force)
            normal = abs(float(contact_force[0]))
            tangent = float(np.linalg.norm(contact_force[1:3]))
            normal_force += normal
            tangential_force += tangent
            support_contacts += 1

            contact_frame = contact.frame.reshape(3, 3)
            vertical_component = abs(float(contact_frame[0, 2])) * normal
            if vertical_component > 0:
                zmp_x += float(contact.pos[0]) * vertical_component
                zmp_y += float(contact.pos[1]) * vertical_component
                vertical_load += vertical_component

        zmp = (
            (zmp_x / vertical_load, zmp_y / vertical_load)
            if vertical_load > 0
            else (0.0, 0.0)
        )
        qvel = d.qvel
        actuator_force = d.actuator_force
        robot_mass = float(m.body_subtreemass[self.robot_root])
        gravity = float(np.linalg.norm(m.opt.gravity))

        return {
            'nq': int(m.nq),
            'nv': int(m.nv),
            'nu': int(m.nu),
            'nbody': int(m.nbody),
            'mass': robot_mass,
            'grav': gravity,
            'torso_z': float(d.xpos[self.torso][2]),
            'tilt': float(tilt),
            'com': d.subtree_com[self.robot_root].copy(),
            'com_v': float(np.linalg.norm(d.subtree_linvel[self.robot_root])),
            'angmom': float(
                np.linalg.norm(d.subtree_angmom[self.robot_root])
            ),
            'ke': float(d.energy[1]),
            'pe': float(d.energy[0]),
            'power': float(np.dot(d.qfrc_actuator, qvel)),
            'maxtq': (
                float(np.max(np.abs(actuator_force)))
                if actuator_force.size
                else 0.0
            ),
            'qvel_rms': (
                float(np.sqrt(np.mean(np.square(qvel)))) if qvel.size else 0.0
            ),
            'ncon': int(d.ncon),
            'support_contacts': support_contacts,
            'fn': normal_force,
            'gravity_comp': (
                normal_force / (robot_mass * gravity)
                if robot_mass > 0 and gravity > 0
                else 0.0
            ),
            'zmp': zmp,
            'slip': tangential_force / (normal_force + 1e-6),
        }
