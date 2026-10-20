"""Time-awareness and opt-in proactive-response decisions."""

import time


class TimeSense:
    def __init__(self):
        self.last_interaction = time.monotonic()

    def update_interaction(self):
        self.last_interaction = time.monotonic()

    def time_since_last_interaction(self):
        return max(0.0, time.monotonic() - self.last_interaction)

    def get_time_of_day_greeting(self, hour=None):
        current_hour = time.localtime().tm_hour if hour is None else int(hour)
        if not 0 <= current_hour <= 23:
            raise ValueError('hour must be in [0, 23]')
        if 5 <= current_hour < 12:
            return 'morning'
        if 12 <= current_hour < 17:
            return 'afternoon'
        if 17 <= current_hour < 22:
            return 'evening'
        return 'night'


class ProactiveInitiation:
    """Suggests a prompt; the caller decides whether unsolicited speech is enabled."""

    def __init__(
        self,
        time_sense: TimeSense,
        idle_threshold: float = 3600.0,
        cooldown: float = 300.0,
    ):
        self.time_sense = time_sense
        self.idle_threshold = idle_threshold
        self.cooldown = cooldown
        self.last_proactive_speech = 0.0

    def should_initiate(self, is_idle=True):
        now = time.monotonic()
        if not is_idle or now - self.last_proactive_speech < self.cooldown:
            return False, ''
        if self.time_sense.time_since_last_interaction() >= self.idle_threshold:
            self.last_proactive_speech = now
            return True, "It's been a while. Is there anything you'd like to talk about?"
        return False, ''


class SelfAwareness:
    def __init__(self):
        self.confidence = 1.0

    def update_confidence(self, perception_confidence, memory_match_score):
        values = (perception_confidence, memory_match_score)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError('Confidence values must be in [0, 1]')
        self.confidence = 0.5 * perception_confidence + 0.5 * memory_match_score

    def get_uncertainty_prefix(self):
        if self.confidence < 0.4:
            return "I'm not entirely sure, but "
        if self.confidence < 0.7:
            return 'I think '
        return ''

    def acknowledge_mistake(self):
        self.confidence = max(0.5, self.confidence)
        return 'Sorry, let me try that again.'
