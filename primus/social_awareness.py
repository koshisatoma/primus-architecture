import re
import time

from primus.detect import PERSON, boxes


ADDRESS_PATTERN = re.compile(r'\b(?:primus|robot)\b', re.IGNORECASE)
PERSON_CHECK_INTERVAL = 2.0


class SocialAwareness:
    """Require an explicit wake word when multiple people are visible."""

    def __init__(self):
        self._last_check = 0.0
        self._person_count = 0

    @staticmethod
    def is_addressed(text):
        return bool(ADDRESS_PATTERN.search(text))

    def should_respond(self, text, frame, interaction_active=False):
        if self.is_addressed(text) or interaction_active:
            return True

        now = time.monotonic()
        if frame is None:
            self._person_count = 0
            self._last_check = now
        elif now - self._last_check >= PERSON_CHECK_INTERVAL:
            self._person_count = sum(
                detection[0] == PERSON for detection in boxes(frame)
            )
            self._last_check = now

        return self._person_count < 2
