import threading


_lock = threading.Lock()
_teacher_frame = None


def publish_teacher_frame(frame):
    global _teacher_frame
    with _lock:
        _teacher_frame = frame


def latest_teacher_frame():
    with _lock:
        return _teacher_frame
