"""Idle-time play suggestions, deliberately disconnected from robot control."""

import random
import time

import numpy as np


class PlayStateMachine:
    def __init__(self, spatial_memory, rng=None):
        self.spatial_memory = spatial_memory
        self.state = 'idle'
        self.last_play = time.monotonic()
        self.play_duration = 0.0
        self.target_pos = None
        self._rng = rng or random.Random()

    def tick(self, is_interacting, is_asleep, dt_hours=0.0):
        if not np.isfinite(dt_hours) or dt_hours < 0:
            raise ValueError('dt_hours must be finite and non-negative')
        if is_interacting or is_asleep:
            self.state = 'idle'
            self.target_pos = None
            self.play_duration = 0.0
            return None
        now = time.monotonic()
        if self.state == 'idle':
            if now - self.last_play > 120 and self._rng.random() < 0.1:
                self._initiate_play()
        else:
            self.play_duration -= dt_hours * 3600
            if self.play_duration <= 0:
                self.state = 'idle'
                self.target_pos = None
                self.last_play = now
                return None
        if self.state == 'object_play':
            return self._suggest_object_play()
        if self.state == 'locomotor_play':
            return self._suggest_locomotor_play()
        return None

    def _initiate_play(self):
        objects = self.spatial_memory.get_obstacles()
        if objects and self._rng.random() < 0.6:
            self.state = 'object_play'
            self.target_pos = self._rng.choice(objects)
            self.play_duration = self._rng.uniform(10, 30)
        else:
            self.state = 'locomotor_play'
            self.target_pos = np.array(
                [self._rng.uniform(-2, 2), self._rng.uniform(-2, 2), 0.0]
            )
            self.play_duration = self._rng.uniform(15, 45)

    def _suggest_object_play(self):
        return {
            'type': 'object_play_suggestion',
            'target': np.asarray(self.target_pos).copy(),
            'actuated': False,
        }

    def _suggest_locomotor_play(self):
        return {
            'type': 'locomotor_play_suggestion',
            'target': np.asarray(self.target_pos).copy(),
            'velocity_scale': 0.5,
            'actuated': False,
        }
