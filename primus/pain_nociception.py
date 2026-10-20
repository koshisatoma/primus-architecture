"""Bounded simulated nociception and avoidance features (not hardware safety)."""

import math
import time

import numpy as np


class PainSystem:
    def __init__(self, force_threshold=50.0):
        if not math.isfinite(float(force_threshold)) or force_threshold < 0:
            raise ValueError('force_threshold must be finite and non-negative')
        self.force_threshold = float(force_threshold)
        self.acute_pain = 0.0
        self.acute_stress = 0.0
        self.chronic_stress = 0.0
        self.avoidance = {}
        self._last = None

    def nocicept(self, impact_force):
        if not math.isfinite(float(impact_force)) or impact_force < 0:
            raise ValueError('impact_force must be finite and non-negative')
        if impact_force <= self.force_threshold:
            return 0.0
        base = min(1.0, (impact_force - self.force_threshold) / 100.0)
        return min(1.0, base * (1.0 + 0.5 * self.chronic_stress))

    def register_pain(self, pain, cue_key=None, store=None, context=None):
        if not math.isfinite(float(pain)) or not 0 <= pain <= 1:
            raise ValueError('pain must be finite and in [0, 1]')
        if pain <= 0.05:
            return None
        self.acute_pain = min(1.0, self.acute_pain + pain)
        self.acute_stress = min(1.0, self.acute_stress + 0.3 * pain)
        if cue_key is not None:
            self.avoidance[cue_key] = min(
                1.0, self.avoidance.get(cue_key, 0.0) + 0.5 * pain
            )
        if store is not None:
            store.store(
                'episodic',
                {
                    'event': 'simulated_contact',
                    'cue': cue_key,
                    'context': context,
                    'intensity': pain,
                },
                salience=0.6 + 0.4 * pain,
                emotional_valence=-pain,
            )
        return pain

    def protective_reflex(self, contact_pos, ee_pos):
        """Return an advisory withdrawal vector; never sends actuator commands."""
        if self.acute_pain <= 0.05:
            return None
        contact = np.asarray(contact_pos, dtype=np.float64)
        end_effector = np.asarray(ee_pos, dtype=np.float64)
        if (
            contact.shape != (3,)
            or end_effector.shape != (3,)
            or not np.isfinite(contact).all()
            or not np.isfinite(end_effector).all()
        ):
            raise ValueError('contact_pos and ee_pos must be finite 3D vectors')
        direction = end_effector - contact
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        return direction / norm * self.acute_pain

    def avoidance_vector(self, pos, cue_positions):
        position = np.asarray(pos, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('pos must be a finite 3D vector')
        vector = np.zeros(3)
        for key, strength in self.avoidance.items():
            if strength < 0.1 or key not in cue_positions:
                continue
            cue = np.asarray(cue_positions[key], dtype=np.float64)
            if cue.shape != (3,) or not np.isfinite(cue).all():
                raise ValueError(f'cue position for {key!r} must be a finite 3D vector')
            delta = position - cue
            distance = float(np.linalg.norm(delta))
            if 1e-6 < distance < 1.0:
                vector += delta / distance * strength / distance**2
        norm = float(np.linalg.norm(vector))
        if norm > 1.0:
            vector /= norm
        return vector

    def tick(self, dt_hours=None, resting=False, failures=0):
        if dt_hours is None:
            now = time.monotonic()
            elapsed = 0.0 if self._last is None else (now - self._last) / 3600.0
            self._last = now
        else:
            elapsed = float(dt_hours)
        if (
            not math.isfinite(elapsed)
            or elapsed < 0
            or not math.isfinite(float(failures))
            or failures < 0
        ):
            raise ValueError('dt_hours and failures must be finite and non-negative')
        self.acute_pain *= math.exp(-6.0 * elapsed)
        self.acute_stress *= math.exp(-2.0 * elapsed)
        load = 0.1 * failures + 0.2 * self.acute_pain
        if resting:
            self.chronic_stress = max(0.0, self.chronic_stress - 0.15 * elapsed)
        else:
            self.chronic_stress = min(1.0, self.chronic_stress + load * elapsed)
        decay = math.exp(-0.05 * elapsed)
        for key in list(self.avoidance):
            self.avoidance[key] *= decay
            if self.avoidance[key] < 0.05:
                del self.avoidance[key]

    def yerkes_dodson(self, arousal):
        if not math.isfinite(float(arousal)):
            raise ValueError('arousal must be finite')
        level = min(1.0, max(0.0, float(arousal)))
        performance = 1.0 - 2.0 * (level - 0.5) ** 2
        return max(0.3, performance - 0.3 * self.chronic_stress)
