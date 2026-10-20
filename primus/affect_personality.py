"""Bounded affect and configurable conversation-style helpers."""

import math
import random
import time


class EmotionSystem:
    def __init__(self):
        self.valence = 0.0
        self.arousal = 0.5
        self.last_decay = time.monotonic()

    def update_from_interaction(self, valence_delta, arousal_delta):
        if not math.isfinite(valence_delta) or not math.isfinite(arousal_delta):
            raise ValueError('Emotion deltas must be finite')
        self.valence = max(-1.0, min(1.0, self.valence + valence_delta))
        self.arousal = max(0.0, min(1.0, self.arousal + arousal_delta))

    def decay_towards_baseline(self, now=None):
        current = time.monotonic() if now is None else float(now)
        elapsed = max(0.0, current - self.last_decay)
        self.last_decay = current
        factor = math.exp(-0.05 * elapsed)
        self.valence *= factor
        self.arousal = 0.5 + (self.arousal - 0.5) * factor

    def get_memory_salience_boost(self):
        return abs(self.valence) * 0.3 + self.arousal * 0.2

    def get_tts_prosody_modifiers(self):
        return {
            'pitch': 1.0 + self.valence * 0.1 + self.arousal * 0.1,
            'rate': 1.0 + self.arousal * 0.2,
        }


class Personality:
    TRAITS = ('O', 'C', 'E', 'A', 'N')

    def __init__(self, traits=None):
        defaults = {'O': 0.6, 'C': 0.5, 'E': 0.5, 'A': 0.7, 'N': 0.3}
        if traits:
            unknown = set(traits).difference(self.TRAITS)
            if unknown:
                raise ValueError(f'Unknown personality traits: {sorted(unknown)}')
            defaults.update(traits)
        if any(not 0 <= value <= 1 for value in defaults.values()):
            raise ValueError('Personality trait scores must be in [0, 1]')
        self.traits = defaults

    def update(self, trait, delta):
        if trait not in self.traits:
            raise ValueError(f'Unknown personality trait: {trait}')
        if not math.isfinite(delta):
            raise ValueError('Trait delta must be finite')
        self.traits[trait] = max(
            0.0, min(1.0, self.traits[trait] + delta * 0.01)
        )

    def get_system_prompt_modifier(self):
        modifiers = []
        if self.traits['E'] > 0.7:
            modifiers.append('be talkative and enthusiastic')
        if self.traits['A'] > 0.7:
            modifiers.append('be polite and accommodating')
        if self.traits['O'] > 0.7:
            modifiers.append('be curious and ask useful follow-up questions')
        if self.traits['C'] < 0.3:
            modifiers.append('be casual and relaxed')
        if self.traits['N'] > 0.7:
            modifiers.append('be cautious and ask for clarification')
        return ', '.join(modifiers) if modifiers else 'be natural and helpful'


class HumorAndRegister:
    def __init__(self, personality: Personality, emotion: EmotionSystem, rng=None):
        self.personality = personality
        self.emotion = emotion
        self._rng = rng or random.Random()

    def should_joke(self):
        score = (
            self.personality.traits['E'] * 0.5
            + self.emotion.valence * 0.3
            + self.emotion.arousal * 0.2
        )
        return self._rng.random() < max(0.0, min(0.15, score * 0.15))

    def get_register(self):
        if self.emotion.arousal < 0.2:
            return 'brief and calm'
        if self.personality.traits['A'] > 0.8:
            return 'polite'
        return 'casual and friendly'
