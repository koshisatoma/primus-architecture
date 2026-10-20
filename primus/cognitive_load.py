"""Working-memory, agency, metacognition, and practice-state prototypes."""

import math
import time

import numpy as np


class WorkingMemory:
    """A bounded, salience-ranked buffer representing roughly four chunks."""

    CAPACITY = 4

    def __init__(self, capacity=CAPACITY):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self.capacity = capacity
        self.slots = []

    def hold(self, content, salience=0.5):
        if not math.isfinite(float(salience)):
            raise ValueError('salience must be finite')
        self.slots.append(
            {'content': content, 'salience': min(1.0, max(0.0, salience)), 't': time.monotonic()}
        )
        while len(self.slots) > self.capacity:
            self.slots.pop(min(range(len(self.slots)), key=lambda i: self.slots[i]['salience']))

    def chunkify(self, items, label):
        self.hold({'chunk': label, 'items': list(items)}, salience=0.6)

    def load(self):
        return min(1.0, len(self.slots) / self.capacity)

    def free_slots(self, count):
        if not isinstance(count, int) or count < 0:
            raise ValueError('count must be a non-negative integer')
        for _ in range(min(count, len(self.slots))):
            self.slots.pop(min(range(len(self.slots)), key=lambda i: self.slots[i]['salience']))

    def dual_task_cost(self, n_tasks):
        if not isinstance(n_tasks, int) or n_tasks < 1:
            raise ValueError('n_tasks must be a positive integer')
        if n_tasks == 1:
            return 1.0
        return 1.0 / (1.0 + 0.5 * (n_tasks - 1) * self.load())


class SenseOfAgency:
    """Compare predicted and observed outcomes; predictions are optional."""

    def __init__(self, world_model=None, threshold=0.3):
        if not math.isfinite(float(threshold)) or threshold < 0:
            raise ValueError('threshold must be finite and non-negative')
        self.wm = world_model
        self.threshold = float(threshold)
        self.last_error = 0.0

    def predict(self, z, action):
        if self.wm is None:
            return None
        import torch

        with torch.no_grad():
            latent_action = torch.cat([z, action], dim=-1)
            mean, _ = self.wm.plat(latent_action).chunk(2, dim=-1)
            return mean

    def compare(self, predicted, actual, self_acting):
        expected = np.asarray(predicted, dtype=np.float64)
        observed = np.asarray(actual, dtype=np.float64)
        if (
            expected.shape != observed.shape
            or expected.size == 0
            or not np.isfinite(expected).all()
            or not np.isfinite(observed).all()
        ):
            raise ValueError('predicted and actual must be matching finite arrays')
        error = float(np.linalg.norm(expected - observed))
        self.last_error = error
        if not self_acting:
            return {
                'agency': False,
                'external_event': error > self.threshold,
                'error': error,
            }
        if error <= self.threshold:
            return {'agency': True, 'external_event': False, 'error': error}
        return {
            'agency': False,
            'external_event': False,
            'internal_error': True,
            'error': error,
        }

    def error_signal(self):
        return self.last_error > self.threshold


class Metacognition:
    def __init__(self):
        self.calibration = 0.5

    def feeling_of_knowing(self, cue, memory_store=None, wm_load=0.0):
        if not math.isfinite(float(wm_load)):
            raise ValueError('wm_load must be finite')
        load = min(1.0, max(0.0, wm_load))
        base = 0.2
        if memory_store is not None:
            hits = memory_store.recall('semantic', query=str(cue), limit=20, drift=0)
            base = min(0.9, 0.2 + 0.2 * len(hits))
        return max(0.05, base * (1.0 - 0.4 * load) * (0.7 + 0.6 * self.calibration))

    def judgment_of_learning(self, salience, wm_load):
        if not all(math.isfinite(float(value)) for value in (salience, wm_load)):
            raise ValueError('salience and wm_load must be finite')
        return max(0.05, min(0.95, salience * (1.0 - 0.5 * min(1.0, max(0.0, wm_load)))))

    def choose_strategy(self, intent, fok):
        if not math.isfinite(float(fok)):
            raise ValueError('fok must be finite')
        if fok > 0.6:
            return 'recall'
        return 'demonstration' if intent in ('perform', 'action') else 'verbal'

    def update_calibration(self, predicted, actual):
        if not all(math.isfinite(float(value)) for value in (predicted, actual)):
            raise ValueError('predicted and actual must be finite')
        self.calibration = min(
            0.9, max(0.1, self.calibration + 0.1 * (actual - predicted))
        )


class SkillAutomaticity:
    def __init__(self):
        self.practice = {}

    def record(self, skill, success=True):
        if not skill:
            raise ValueError('skill must be non-empty')
        if success:
            self.practice[skill] = self.practice.get(skill, 0) + 1

    def stage(self, skill):
        count = self.practice.get(skill, 0)
        if count < 5:
            return 'cognitive'
        if count < 25:
            return 'associative'
        return 'autonomous'

    def wm_cost(self, skill):
        return {'cognitive': 2, 'associative': 1, 'autonomous': 0}[self.stage(skill)]

    def speed_multiplier(self, skill):
        return {'cognitive': 0.7, 'associative': 1.0, 'autonomous': 1.3}[self.stage(skill)]


class CognitiveLoadManager:
    def __init__(self, world_model=None):
        self.wm = WorkingMemory()
        self.agency = SenseOfAgency(world_model)
        self.meta = Metacognition()
        self.auto = SkillAutomaticity()

    def effective_load(self, active_skills=()):
        load = self.wm.load()
        for skill in active_skills:
            if self.auto.stage(skill) == 'autonomous':
                load = max(0.0, load - 0.25)
        return load

    def performance_factor(self, n_tasks=1, active_skills=()):
        return self.wm.dual_task_cost(n_tasks) * (
            1.0 - 0.3 * self.effective_load(active_skills)
        )
