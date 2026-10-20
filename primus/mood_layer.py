"""Toy affect baseline and fatigue dynamics for offline experiments."""

import math
import time


class MoodLayer:
    def __init__(self, tau_hours=3.0):
        if not math.isfinite(float(tau_hours)) or tau_hours <= 0:
            raise ValueError('tau_hours must be finite and positive')
        self.mood = 0.0
        self.tau = float(tau_hours) * 3600.0
        self._last = time.monotonic()

    def tick(self, emotion_valence=0.0, dt_hours=None):
        now = time.monotonic()
        elapsed = (now - self._last) / 3600 if dt_hours is None else float(dt_hours)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('dt_hours must be finite and non-negative')
        if not math.isfinite(float(emotion_valence)):
            raise ValueError('emotion_valence must be finite')
        self._last = now
        alpha = 1 - math.exp(-elapsed * 3600 / self.tau)
        target = min(1.0, max(-1.0, float(emotion_valence)))
        self.mood = min(1.0, max(-1.0, self.mood + alpha * (target - self.mood)))
        return self.mood

    def event(self, valence_delta):
        if not math.isfinite(float(valence_delta)):
            raise ValueError('valence_delta must be finite')
        self.mood = min(1.0, max(-1.0, self.mood + valence_delta))

    def recall_bias(self):
        return self.mood

    def sociability(self):
        return 0.5 + 0.4 * self.mood

    def perception_note(self, text):
        if self.mood < -0.4 and '?' not in text and len(text) < 60:
            return text + ' [interpretation may be biased by simulated mood]'
        return text


class FatigueModel:
    def __init__(self):
        self.fatigue = 0.0
        self.vigilance = 1.0
        self.time_on_task = 0.0

    def work(self, dt_hours=0.0, load=1.0):
        if (
            not math.isfinite(float(dt_hours))
            or dt_hours < 0
            or not math.isfinite(float(load))
            or load < 0
        ):
            raise ValueError('dt_hours and load must be finite and non-negative')
        self.time_on_task += dt_hours
        self.fatigue = min(1.0, self.fatigue + 0.10 * load * dt_hours)
        self.vigilance = max(
            0.3, 1.0 - 0.5 * max(0.0, self.time_on_task - 0.5)
        )
        return self.fatigue

    def break_rest(self, dt_hours=0.25):
        if not math.isfinite(float(dt_hours)) or dt_hours < 0:
            raise ValueError('dt_hours must be finite and non-negative')
        self.fatigue = max(0.0, self.fatigue - 0.3 * dt_hours)
        self.vigilance = min(1.0, self.vigilance + 0.4 * dt_hours)
        self.time_on_task = 0.0

    def sleep_recovery(self, quality=1.0):
        if not 0 <= quality <= 1:
            raise ValueError('quality must be in [0, 1]')
        self.fatigue = max(0.0, self.fatigue - 0.8 * quality)
        self.vigilance = 1.0
        self.time_on_task = 0.0

    def lapse_probability(self):
        return min(0.5, 0.3 * (1 - self.vigilance) + 0.2 * self.fatigue)

    def performance_multiplier(self):
        return max(0.4, 1.0 - 0.4 * self.fatigue - 0.2 * (1 - self.vigilance))


def combined_performance(sleep_cycle, fatigue, pain_system, arousal=0.5):
    return (
        sleep_cycle.performance_multiplier()
        * fatigue.performance_multiplier()
        * pain_system.yerkes_dodson(arousal)
    )
