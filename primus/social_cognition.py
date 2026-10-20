"""Lightweight social-attention and audio-feature helpers."""

import time

import numpy as np


class AudioTurnTaking:
    """A short-window speech-energy heuristic; not speaker diarization."""

    def __init__(
        self,
        energy_threshold: float = 0.02,
        history_seconds: float = 5.0,
        speech_fraction: float = 0.6,
    ):
        if energy_threshold < 0 or history_seconds <= 0:
            raise ValueError('energy_threshold must be non-negative and window positive')
        if not 0 <= speech_fraction <= 1:
            raise ValueError('speech_fraction must be in [0, 1]')
        self.energy_threshold = energy_threshold
        self.history_seconds = history_seconds
        self.speech_fraction = speech_fraction
        self.speaking_history = []

    def update(self, audio_chunk_rms: float, timestamp: float | None = None):
        if not np.isfinite(audio_chunk_rms):
            raise ValueError('audio_chunk_rms must be finite')
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(now):
            raise ValueError('timestamp must be finite')
        self.speaking_history.append((now, audio_chunk_rms > self.energy_threshold))
        self.speaking_history = [
            (seen_at, speaking)
            for seen_at, speaking in self.speaking_history
            if 0 <= now - seen_at <= self.history_seconds
        ]

    def humans_conversing(self, num_people_detected: int) -> bool:
        if num_people_detected < 2 or not self.speaking_history:
            return False
        speech_count = sum(speaking for _, speaking in self.speaking_history)
        return speech_count / len(self.speaking_history) > self.speech_fraction


class TheoryOfMind:
    """A time-limited visual-attention cue, not an intent detector."""

    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout
        self.attention_target = None
        self.last_update = None

    def update_gaze(self, head_yaw: float, detected_objects):
        if not np.isfinite(head_yaw):
            raise ValueError('head_yaw must be finite')
        now = time.monotonic()
        candidates = [
            (name, float(center))
            for name, center in detected_objects
            if isinstance(name, str) and name and np.isfinite(center)
        ]
        self.last_update = now
        self.attention_target = (
            min(candidates, key=lambda item: abs(item[1] - head_yaw))[0]
            if candidates
            else None
        )
        return self.attention_target

    def get_primed_object(self):
        if (
            self.last_update is not None
            and time.monotonic() - self.last_update < self.timeout
        ):
            return self.attention_target
        return None


class EmotionRecognition:
    """Returns simple acoustic features; they are not reliable emotion labels."""

    def analyze_prosody(self, audio_chunk):
        values = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return {'arousal': 0.0, 'valence': 0.0}
        if not np.isfinite(values).all():
            raise ValueError('audio_chunk must contain finite values')
        rms = float(np.sqrt(np.mean(np.square(values))))
        signs = np.signbit(values)
        zero_crossings = float(np.count_nonzero(signs[1:] != signs[:-1]))
        zcr = zero_crossings / max(1, values.size - 1)
        return {
            'arousal': float(np.clip(rms * 10.0, 0.0, 1.0)),
            # This is only a weak acoustic proxy, not inferred emotional valence.
            'valence': 0.0,
            'zero_crossing_rate': zcr,
        }
