import mujoco
import numpy as np
from primus.scene_builder import load_model


class Sim:
    def __init__(self, h=540, w=960):
        self.m = load_model()
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat = np.array([0.0, 0.6, 0.9])
        self.cam.distance = 3.5
        mujoco.mj_forward(self.m, self.d)

    def named(self, name):
        self.r.update_scene(self.d, name)
        return self.r.render()[..., ::-1]

    def render_three(self, azimuth=200, elevation=-15, distance=3.5):
        return (
            self.orbit(azimuth, elevation, distance),
            self.named('front3m'),
            self.named('corner'),
        )

    def orbit(self, azimuth, elevation, distance):
        self.cam.azimuth, self.cam.elevation, self.cam.distance = azimuth, elevation, distance
        self.r.update_scene(self.d, self.cam)
        return self.r.render()[..., ::-1]

    def close(self):
        self.r.close()