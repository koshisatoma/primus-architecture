"""Generate symbolic nonverbal suggestions; outputs are never actuator targets."""

import math
import random


class NonverbalGenerator:
    def __init__(self, rng=None):
        self.current_gesture = None
        self.gesture_duration = 0.0
        self._rng = rng or random.Random()

    def get_gesture(self, intent, emotion_valence, emotion_arousal):
        if self.gesture_duration > 0:
            return self.current_gesture
        if intent == 'greeting':
            self.current_gesture, self.gesture_duration = 'wave', 1.5
        elif intent in ('pointing', 'grounding'):
            self.current_gesture, self.gesture_duration = 'point', 2.0
        elif intent == 'affirmation':
            self.current_gesture, self.gesture_duration = 'nod', 0.8
        elif intent == 'negation':
            self.current_gesture, self.gesture_duration = 'shake_head', 0.8
        elif emotion_valence > 0.5 and emotion_arousal > 0.6:
            self.current_gesture = (
                'clap' if self._rng.random() < 0.3 else 'idle_hands'
            )
            self.gesture_duration = 1.0 if self.current_gesture == 'clap' else 0.0
        else:
            self.current_gesture, self.gesture_duration = 'idle_hands', 0.0
        return self.current_gesture

    def tick(self, dt):
        self.gesture_duration = max(0.0, self.gesture_duration - dt)
        if self.gesture_duration == 0:
            self.current_gesture = None

    def get_posture_offsets(self, emotion_valence, fatigue):
        if (
            not math.isfinite(float(emotion_valence))
            or not math.isfinite(float(fatigue))
            or not -1 <= emotion_valence <= 1
            or not 0 <= fatigue <= 1
        ):
            raise ValueError('emotion_valence and fatigue are outside their bounds')
        valence = min(1.0, max(-1.0, emotion_valence))
        tiredness = min(1.0, max(0.0, fatigue))
        shoulder_slump = -0.1 * max(0.0, -valence)
        return {
            'torso_pitch': 0.1 * valence - 0.15 * tiredness,
            'head_pitch': -0.2 * max(0.0, -valence) - 0.2 * tiredness,
            'left_shoulder_pitch': shoulder_slump,
            'right_shoulder_pitch': shoulder_slump,
        }

    def get_gaze_aversion(self, cognitive_load):
        if not math.isfinite(float(cognitive_load)) or not 0 <= cognitive_load <= 1:
            raise ValueError('cognitive_load must be in [0, 1]')
        return (self._rng.choice((-0.3, 0.3)), 0.2) if cognitive_load > 0.7 else (0, 0)
