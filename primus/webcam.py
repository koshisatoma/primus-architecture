import logging
import threading

import cv2


LOGGER = logging.getLogger(__name__)
WIDTH = 960
HEIGHT = 540


class Webcam:
    def __init__(self, index=0):
        self.frame = None
        self.ok = False
        self.status = 'Connecting to teacher camera'
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, args=(index,), daemon=True, name='primus-webcam'
        )
        self._thread.start()

    def _loop(self, index):
        cap = cv2.VideoCapture(index)
        try:
            self.ok = cap.isOpened()
            if not self.ok:
                self.status = f'Camera {index} is unavailable'
                LOGGER.warning(self.status)
                return

            self.status = f'Teacher camera {index} is live'
            while self.ok and not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.ok = False
                    self.status = f'Camera {index} stopped returning frames'
                    LOGGER.warning(self.status)
                    break
                frame = self._fit_viewport(frame)
                with self._lock:
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.ok = False

    @staticmethod
    def _fit_viewport(frame):
        height, width = frame.shape[:2]
        target_ratio = WIDTH / HEIGHT
        frame_ratio = width / height
        if frame_ratio > target_ratio:
            crop_width = round(height * target_ratio)
            start = (width - crop_width) // 2
            frame = frame[:, start:start + crop_width]
        else:
            crop_height = round(width / target_ratio)
            start = (height - crop_height) // 2
            frame = frame[start:start + crop_height, :]
        return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    def get(self):
        with self._lock:
            return self.frame
