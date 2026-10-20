"""Toy developmental-learning state; not a model of human development."""

import random
import time
import math


class Temperament:
    TRAITS = ('O', 'C', 'E', 'A', 'N')

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self.activity = self._rng.uniform(0.2, 0.8)
        self.soothability = self._rng.uniform(0.2, 0.8)
        self.sociability = self._rng.uniform(0.2, 0.8)
        self.fearfulness = self._rng.uniform(0.1, 0.6)

    def personality_init(self):
        return {
            'O': 0.3 + 0.4 * self.activity,
            'C': 0.5,
            'E': self.sociability,
            'A': self.soothability,
            'N': self.fearfulness,
        }


class SensitivePeriods:
    def __init__(self, birth=None):
        self.birth = time.monotonic() if birth is None else float(birth)
        if not math.isfinite(self.birth):
            raise ValueError('birth must be finite')
        self.windows = {
            'language': (0.0, 720.0),
            'attachment': (0.0, 1440.0),
            'motor': (0.0, 4320.0),
        }

    def multiplier(self, domain):
        low, high = self.windows.get(domain, (0.0, 1e9))
        age_hours = max(0.0, (time.monotonic() - self.birth) / 3600.0)
        if age_hours <= low:
            return 1.0
        if age_hours >= high:
            return 0.3
        return 1.0 - 0.7 * (age_hours - low) / (high - low)


class LanguageAcquisition:
    OVERGENERALIZATION = {
        'go': 'goed',
        'went': 'goed',
        'eat': 'eated',
        'ate': 'eated',
    }

    def __init__(self, rng=None):
        self.vocab = set()
        self._rng = rng or random.Random()

    def hear(self, word):
        if not isinstance(word, str) or not word.strip():
            raise ValueError('word must be a non-empty string')
        self.vocab.add(word.casefold())

    def stage(self):
        count = len(self.vocab)
        if count < 10:
            return 'babbling'
        if count < 50:
            return 'one-word'
        if count < 200:
            return 'two-word'
        return 'telegraphic'

    def produce(self, intent_words):
        stage = self.stage()
        output = []
        for word in intent_words:
            if (
                stage in ('two-word', 'telegraphic')
                and word in self.OVERGENERALIZATION
                and self._rng.random() < 0.25
            ):
                word = self.OVERGENERALIZATION[word]
            output.append(word)
        return ' '.join(output[:2] if stage == 'two-word' else output)


class NormativeLearning:
    def __init__(self):
        self.norms = {}

    def correction(self, norm_key, valence=-0.5):
        if not norm_key:
            raise ValueError('norm_key must be non-empty')
        if (
            not isinstance(valence, (int, float))
            or not math.isfinite(valence)
            or valence > 0
        ):
            raise ValueError('correction valence must be numeric and non-positive')
        self.norms[norm_key] = self.norms.get(norm_key, 0.0) + valence

    def violation_reaction(self, norm_key):
        if self.norms.get(norm_key, 0.0) < -0.5:
            return f'That may violate our learned norm: {norm_key}.'
        return None
