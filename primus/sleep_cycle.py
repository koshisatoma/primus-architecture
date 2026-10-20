"""A deterministic sleep-state simulation for offline cognition experiments."""

import math
import random
import time

from primus.memory_store import memory_store


class SleepCycle:
    """Toy homeostatic/circadian model; it never gates the live voice agent."""

    STATES = {'awake', 'nrem', 'rem'}

    def __init__(self, rng=None):
        self.state = 'awake'
        self.sleep_pressure = 0.0
        self.sleep_debt = 0.0
        self.inertia = 0.0
        self.cycle = 0
        self.dream_log = []
        self._last = time.monotonic()
        self._rng = rng or random.Random()

    def circadian_drive(self, hour=None):
        local_time = time.localtime()
        current_hour = (
            local_time.tm_hour + local_time.tm_min / 60.0
            if hour is None else float(hour)
        )
        if not math.isfinite(current_hour):
            raise ValueError('hour must be finite')
        current_hour %= 24.0
        return 0.5 * (1 + math.cos(2 * math.pi * (current_hour - 3) / 24))

    def tick(self, dt_hours=None):
        now = time.monotonic()
        elapsed = (now - self._last) / 3600.0 if dt_hours is None else float(dt_hours)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('dt_hours must be finite and non-negative')
        self._last = now
        if self.state == 'awake':
            self.sleep_pressure = min(1.0, self.sleep_pressure + 0.08 * elapsed)
            self.inertia = max(0.0, self.inertia - 1.5 * elapsed)
        else:
            self.sleep_pressure = max(0.0, self.sleep_pressure - 0.25 * elapsed)
            self.sleep_debt = max(0.0, self.sleep_debt - 0.1 * elapsed)
        return self.sleepiness()

    def sleepiness(self):
        return self.sleep_pressure + 0.4 * self.circadian_drive() + self.sleep_debt

    def should_sleep(self):
        return self.state == 'awake' and self.sleepiness() > 0.85

    def is_asleep(self):
        return self.state != 'awake'

    def begin_sleep(self):
        if self.state != 'awake':
            return False
        self.state = 'nrem'
        self.cycle += 1
        return True

    def step_cycle(self, store=None, world_model=None):
        del world_model  # No trained, compatible dream model is available.
        store = store or memory_store()
        if self.state == 'nrem':
            consolidated = store.consolidate(threshold=0.65)
            pruned = store.prune_below(0.12, un_consolidated_only=True)
            self.sleep_debt = max(0.0, self.sleep_debt - 0.05)
            self.state = 'rem'
            return {'consolidated': consolidated, 'pruned': pruned}
        if self.state == 'rem':
            dreams = self._memory_dream(store)
            self._rehearse(store)
            self.dream_log.extend(dreams)
            if self.sleep_pressure < 0.2:
                self.wake()
            return {'dreams': dreams}
        return {}

    def _memory_dream(self, store):
        episodes = store.recall('episodic', limit=10, drift=0)
        dict_episodes = [
            item['content'] for item in episodes
            if isinstance(item.get('content'), dict)
        ]
        if len(dict_episodes) < 2:
            return []
        first, second = self._rng.sample(dict_episodes, 2)
        dream = {**first, **{key: value for key, value in second.items() if key not in first}}
        dream['dream'] = True
        store.store('dream', dream, salience=0.2, novelty=0.9)
        return [dream]

    @staticmethod
    def _rehearse(store):
        store.reinforce_kind('procedural', increment=0.02)

    def wake(self, stimulus=0.0):
        if not math.isfinite(float(stimulus)) or stimulus < 0:
            raise ValueError('stimulus must be finite and non-negative')
        if self.state == 'awake':
            return True
        if 0.0 < stimulus < 0.7:
            return False
        self.state = 'awake'
        self.inertia = min(1.0, 0.6 + self.sleep_debt)
        self._last = time.monotonic()
        return True

    def performance_multiplier(self):
        return max(0.4, 1.0 - 0.5 * self.inertia - 0.2 * self.sleep_debt)
