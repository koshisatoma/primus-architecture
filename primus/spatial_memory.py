"""In-process 3D object memory for caller-supplied detections."""

import time

import numpy as np


class SpatialMemory:
    def __init__(self, decay_rate: float = 0.05, merge_threshold: float = 0.15):
        if decay_rate < 0 or merge_threshold <= 0:
            raise ValueError('decay_rate must be non-negative and threshold positive')
        self.decay_rate = float(decay_rate)
        self.merge_threshold = float(merge_threshold)
        self.objects = {}

    def _decay(self, now):
        expired = []
        for name, item in self.objects.items():
            elapsed = max(0.0, now - item['last_seen'])
            item['confidence'] *= float(np.exp(-self.decay_rate * elapsed))
            if item['confidence'] <= 0.05:
                expired.append(name)
        for name in expired:
            del self.objects[name]

    def update(self, detections_3d, timestamp=None):
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(now):
            raise ValueError('timestamp must be finite')
        self._decay(now)
        for name, position, class_id in detections_3d:
            if not isinstance(name, str) or not name.strip():
                raise ValueError('Detection names must be non-empty strings')
            point = np.asarray(position, dtype=np.float64)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('Detection positions must be finite 3D vectors')
            match = next(
                (
                    item for item in self.objects.values()
                    if item['class_id'] == int(class_id)
                    and np.linalg.norm(item['pos'] - point) < self.merge_threshold
                ),
                None,
            )
            if match is not None:
                match['pos'] = 0.7 * match['pos'] + 0.3 * point
                match['confidence'] = min(1.0, match['confidence'] + 0.2)
                match['last_seen'] = now
                match['name'] = name.strip()
            else:
                key = name.strip()
                if key in self.objects:
                    key = f'{key}:{int(class_id)}'
                self.objects[key] = {
                    'name': name.strip(),
                    'pos': point.copy(),
                    'confidence': 0.5,
                    'last_seen': now,
                    'class_id': int(class_id),
                }

    def query(self, name: str):
        for item in self.objects.values():
            if item['name'].casefold() == name.casefold():
                return item['pos'].copy()
        return None

    def get_obstacles(self, min_confidence: float = 0.3):
        return [
            item['pos'].copy()
            for item in self.objects.values()
            if item['confidence'] > min_confidence
        ]

    def get_state_vector(self):
        ordered = sorted(
            self.objects.values(),
            key=lambda item: item['confidence'],
            reverse=True,
        )[:5]
        result = np.zeros(15, dtype=np.float32)
        for index, item in enumerate(ordered):
            result[index * 3:(index + 1) * 3] = item['pos']
        return result
