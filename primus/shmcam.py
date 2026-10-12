import logging
import time
from multiprocessing import resource_tracker, shared_memory

import numpy as np


LOGGER = logging.getLogger(__name__)
HEADER_BYTES = 64
HEADER_ITEMS = HEADER_BYTES // np.dtype(np.uint64).itemsize
SHARED_MEMORY_NAME = 'primus_cam'
CAMERA_FRESHNESS_SECONDS = 2.0
READ_RETRIES = 3
HEADER_RETRIES = 1000


class ShmCam:
    def __init__(self, name=SHARED_MEMORY_NAME):
        self.shm = shared_memory.SharedMemory(name=name, create=False)
        resource_tracker.unregister(self.shm._name, 'shared_memory')
        self.header = np.ndarray(
            (HEADER_ITEMS,), dtype=np.uint64, buffer=self.shm.buf[:HEADER_BYTES]
        )
        self._last_frame = None
        self.status = 'Connected to C++ shared-memory camera'
        try:
            width, height, timestamp_ns = self._read_dimensions()
            if width <= 0 or height <= 0 or self.shm.size < (
                HEADER_BYTES + width * height * 3
            ):
                raise ValueError(
                    'C++ camera shared memory has invalid dimensions'
                )
            if time.time_ns() - timestamp_ns > CAMERA_FRESHNESS_SECONDS * 1e9:
                raise RuntimeError(
                    'C++ camera shared memory has no recent frame'
                )
        except Exception:
            self.close()
            raise
        self.width = width
        self.height = height

    def _read_dimensions(self):
        for _ in range(HEADER_RETRIES):
            before = int(self.header[0])
            if before & 1:
                time.sleep(0)
                continue
            width, height, timestamp_ns = (
                int(self.header[1]),
                int(self.header[2]),
                int(self.header[3]),
            )
            after = int(self.header[0])
            if before == after and not (after & 1):
                return width, height, timestamp_ns
            time.sleep(0)
        raise RuntimeError('C++ camera shared-memory header is being updated')

    def stats(self):
        for _ in range(HEADER_RETRIES):
            before = int(self.header[0])
            if before & 1:
                time.sleep(0)
                continue
            timestamp_ns = int(self.header[3])
            fps = int(self.header[4])
            after = int(self.header[0])
            if before == after and not (after & 1):
                return after // 2, timestamp_ns, fps
            time.sleep(0)
        raise RuntimeError('Could not read a consistent C++ camera header')

    def get(self):
        for _ in range(READ_RETRIES):
            before = int(self.header[0])
            if before == 0 or before & 1:
                continue
            frame_bytes = self.width * self.height * 3
            frame = np.ndarray(
                (self.height, self.width, 3),
                dtype=np.uint8,
                buffer=self.shm.buf[HEADER_BYTES : HEADER_BYTES + frame_bytes],
            ).copy()
            after = int(self.header[0])
            if before == after and not (after & 1):
                self._last_frame = frame
                return frame
        return self._last_frame

    def close(self):
        shm = getattr(self, 'shm', None)
        if shm is not None:
            self.header = None
            shm.close()
            self.shm = None
