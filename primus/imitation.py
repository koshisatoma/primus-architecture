"""Explicit demonstration and event-response helpers; no motor execution."""

import math
import time

import numpy as np


class MirrorImitation:
    """Store named demonstrations and return symbolic, non-actuated copies."""

    def __init__(self, body=None, max_samples=100):
        self.body = body
        self.max_samples = max_samples
        self.library = {}

    def observe(self, gesture_name, observed_joint_traj):
        if not gesture_name:
            raise ValueError('gesture_name must be non-empty')
        values = np.asarray(observed_joint_traj, dtype=np.float32)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError('trajectory must contain finite values')
        self.library[gesture_name] = values.copy()
        while len(self.library) > self.max_samples:
            self.library.pop(next(iter(self.library)))
        return values.copy()

    def reproduce(self, gesture_name, scale=1.0):
        if gesture_name not in self.library:
            return None
        if not math.isfinite(float(scale)):
            raise ValueError('scale must be finite')
        return self.library[gesture_name] * scale

    def novelty(self, gesture_name):
        return 0.0 if gesture_name in self.library else 1.0


class IntuitivePhysics:
    VOE = {'float': 0.9, 'pass_through': 0.8, 'vanish': 0.7}

    def __init__(self, surprise_thr=0.5):
        self.thr = surprise_thr
        self.surprise_log = []

    def violation(self, event):
        score = self.VOE.get(event.get('type'), 0.0)
        if score > self.thr:
            self.surprise_log.append((time.monotonic(), score))
        return score

    def latent_surprise(self, predicted, actual):
        expected = np.asarray(predicted, dtype=np.float64)
        observed = np.asarray(actual, dtype=np.float64)
        if (
            expected.shape != observed.shape
            or expected.size == 0
            or not np.isfinite(expected).all()
            or not np.isfinite(observed).all()
        ):
            raise ValueError('predicted and actual must be matching finite arrays')
        surprise = float(np.linalg.norm(expected - observed))
        if surprise > self.thr:
            self.surprise_log.append((time.monotonic(), surprise))
        return surprise


class IntervalTiming:
    def __init__(self):
        self._start = None
        self._arousal = 0.5

    def start(self, arousal=0.5):
        if not math.isfinite(float(arousal)):
            raise ValueError('arousal must be finite')
        self._start = time.monotonic()
        self._arousal = min(1.0, max(0.0, float(arousal)))

    def estimate(self):
        if self._start is None:
            return 0.0
        elapsed = time.monotonic() - self._start
        return elapsed * (1.0 + 2.0 * max(0.0, self._arousal - 0.5))


class StartleReflex:
    def __init__(self, decay=0.8):
        if not 0 <= decay <= 1:
            raise ValueError('decay must be in [0, 1]')
        self.sensitization = 0.0
        self.decay = decay
        self.weak_count = 0
        self._last = time.monotonic()

    def response(self, stimulus):
        if not math.isfinite(float(stimulus)):
            raise ValueError('stimulus must be finite')
        intensity = min(1.0, max(0.0, float(stimulus)))
        elapsed_minutes = (time.monotonic() - self._last) / 60.0
        self.sensitization *= math.exp(-0.5 * elapsed_minutes)
        self._last = time.monotonic()
        if intensity > 0.8:
            self.sensitization = min(1.0, self.sensitization + 0.6)
            self.weak_count = 0
        elif intensity < 0.3:
            self.weak_count += 1
            return min(
                1.0,
                intensity * math.exp(-0.3 * self.weak_count) * (1 + self.sensitization),
            )
        return min(1.0, intensity * (1 + self.sensitization))


class CocktailParty:
    """Select a named stream only when caller supplies scored candidates."""

    def __init__(self, teacher_pitch=None):
        self.teacher_pitch = teacher_pitch
        self.attended = False

    def update(self, candidates, own_name_heard=False):
        if own_name_heard:
            self.attended = True
            return 'teacher'
        if not candidates or self.teacher_pitch is None:
            self.attended = False
            return None
        valid = [
            (float(pitch), float(energy))
            for pitch, energy in candidates
            if math.isfinite(float(pitch))
            and math.isfinite(float(energy))
            and energy >= 0
        ]
        if not valid:
            self.attended = False
            return None
        _, energy = min(valid, key=lambda source: abs(source[0] - self.teacher_pitch))
        self.attended = energy > 0.02
        return 'teacher' if self.attended else None
