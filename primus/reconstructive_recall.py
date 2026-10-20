"""Optional experimental memory reconstruction helpers.

Drift is disabled by default: do not use reconstructed values as factual
records, object coordinates, names, preferences, or safety-critical inputs.
"""

import copy
import random
from typing import Any


class ReconstructiveRecall:
    def __init__(
        self,
        drift_strength: float = 0.05,
        drift_probability: float = 0.3,
        rng: random.Random | None = None,
    ):
        if drift_strength < 0 or not 0 <= drift_probability <= 1:
            raise ValueError('drift_strength must be non-negative and probability in [0, 1]')
        self.drift_strength = drift_strength
        self.drift_probability = drift_probability
        self._rng = rng or random.Random()

    def recall_with_drift(self, memory: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self._apply_drift(value)
            if self._rng.random() < self.drift_probability
            else copy.deepcopy(value)
            for key, value in memory.items()
        }

    def _apply_drift(self, value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value * (1 + self._rng.gauss(0, self.drift_strength))
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return [self._apply_drift(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._apply_drift(item)
                if self._rng.random() < self.drift_probability
                else copy.deepcopy(item)
                for key, item in value.items()
            }
        return copy.deepcopy(value)

    def recall_episode(
        self, episode: dict[str, Any], drift: bool = False
    ) -> dict[str, Any]:
        return self.recall_with_drift(episode) if drift else copy.deepcopy(episode)


_recall = None


def reconstructive_recall() -> ReconstructiveRecall:
    global _recall
    if _recall is None:
        _recall = ReconstructiveRecall()
    return _recall
