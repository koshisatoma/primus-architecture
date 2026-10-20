"""Small stateful social-learning helpers; inputs must be explicitly supplied."""

import math
import time

import numpy as np


class AttachmentSystem:
    def __init__(self):
        self.bonds = {}
        self.primary_caregiver = None
        self.last_seen = {}

    def interact(self, person_id, valence_delta, duration_hours=0.1):
        if not person_id or not math.isfinite(float(valence_delta)):
            raise ValueError('person_id and finite valence_delta are required')
        if not math.isfinite(float(duration_hours)) or duration_hours < 0:
            raise ValueError('duration_hours must be finite and non-negative')
        self.bonds.setdefault(person_id, 0.1)
        self.bonds[person_id] = min(
            1.0,
            self.bonds[person_id]
            + 0.05 * duration_hours
            + 0.1 * max(0.0, valence_delta),
        )
        self.last_seen[person_id] = time.monotonic()
        if (
            self.primary_caregiver is None
            or self.bonds[person_id] > self.bonds[self.primary_caregiver]
        ):
            self.primary_caregiver = person_id

    def get_stranger_anxiety(self, person_id):
        if person_id not in self.bonds or self.bonds[person_id] < 0.2:
            if self.primary_caregiver is not None:
                last_seen = self.last_seen.get(self.primary_caregiver, 0.0)
                if time.monotonic() - last_seen > 60:
                    return 0.8
            return 0.5
        return 0.0

    def get_separation_distress(self):
        if self.primary_caregiver is None:
            return 0.0
        last_seen = self.last_seen.get(self.primary_caregiver, 0.0)
        elapsed = time.monotonic() - last_seen
        return min(1.0, max(0.0, (elapsed - 300) / 600))

    def get_proxemics_preference(self, person_id):
        bond = self.bonds.get(person_id, 0.1)
        return 1.5 - bond


class EmpathySystem:
    def __init__(self):
        self.contagion_level = 0.0

    def catch_emotion(self, other_valence, other_arousal, bond_strength=0.5):
        values = (other_valence, other_arousal, bond_strength)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError('emotion values and bond strength must be finite')
        self.contagion_level = min(1.0, max(0.0, bond_strength)) * 0.5
        return (
            min(1.0, max(-1.0, other_valence)) * self.contagion_level,
            min(1.0, max(0.0, other_arousal)) * self.contagion_level,
        )

    def should_comfort(self, other_valence, other_arousal):
        if other_valence < -0.4 and other_arousal > 0.3:
            return True, "You may be upset. Would you like support?"
        if other_valence < -0.2 and other_arousal < 0.3:
            return True, "You may be feeling down. Would you like to talk?"
        return False, ''


class FalseBeliefToM:
    """Explicitly tracks observed beliefs separately from observed reality."""

    def __init__(self, distance_threshold=0.2):
        self.teacher_beliefs = {}
        self.reality = {}
        self.distance_threshold = float(distance_threshold)
        if not math.isfinite(self.distance_threshold) or self.distance_threshold < 0:
            raise ValueError('distance_threshold must be finite and non-negative')

    @staticmethod
    def _position(position):
        result = np.asarray(position, dtype=np.float64)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError('position must be a finite 3D vector')
        return result

    def update_reality(self, object_name, position):
        self.reality[object_name] = self._position(position)

    def teacher_sees(self, object_name, position):
        self.teacher_beliefs[object_name] = self._position(position)

    def has_false_belief(self, object_name):
        if object_name not in self.teacher_beliefs or object_name not in self.reality:
            return False
        return (
            np.linalg.norm(
                self.teacher_beliefs[object_name] - self.reality[object_name]
            )
            > self.distance_threshold
        )

    def get_correction_strategy(self, object_name):
        return 'explain_and_point' if self.has_false_belief(object_name) else 'none'
