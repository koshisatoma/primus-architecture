"""Presentation transforms for the live camera dashboard."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from primus import objects
from primus.detect import boxes, class_names


LOGGER = logging.getLogger(__name__)
CHOICES = [
    'Live Camera (RGB)',
    'Depth (Live Camera)',
    'Occupancy (Live Camera)',
    'Gradient (Live Camera)',
    'Detection & Recognition (Live Camera)',
]
_DEPTH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='primus-depth')
_DEPTH_LOCK = threading.Lock()
_DEPTH_FUTURE = None
_DEPTH_RESULT = None
_DEPTH_REQUEST_AFTER = 0.0
DEPTH_REQUEST_INTERVAL = 1.0


def gradient(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(horizontal, vertical)
    magnitude = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)


def _local_depth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    fine = cv2.GaussianBlur(gray, (5, 5), 0)
    coarse = cv2.GaussianBlur(gray, (51, 51), 0)
    estimate = np.abs(fine - coarse) + (1 - gray) * 0.3
    return cv2.normalize(
        estimate, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def _request_depth(jpeg_bytes):
    from primus.cloud_client import depth_cloud

    return depth_cloud(jpeg_bytes)


def _depth(frame):
    global _DEPTH_FUTURE, _DEPTH_RESULT, _DEPTH_REQUEST_AFTER

    with _DEPTH_LOCK:
        if _DEPTH_FUTURE is not None and _DEPTH_FUTURE.done():
            try:
                _DEPTH_RESULT = _DEPTH_FUTURE.result()
            except Exception:
                LOGGER.exception('Cloud depth request failed')
                _DEPTH_RESULT = None
            _DEPTH_FUTURE = None

        now = time.monotonic()
        if _DEPTH_FUTURE is None and now >= _DEPTH_REQUEST_AFTER:
            success, encoded = cv2.imencode(
                '.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            if not success:
                raise RuntimeError('OpenCV could not encode the camera frame for depth')
            _DEPTH_FUTURE = _DEPTH_EXECUTOR.submit(
                _request_depth, encoded.tobytes()
            )
            _DEPTH_REQUEST_AFTER = now + DEPTH_REQUEST_INTERVAL
        estimated_depth = _DEPTH_RESULT

    if estimated_depth is not None and estimated_depth.shape == frame.shape[:2]:
        normalized = cv2.normalize(
            estimated_depth, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        source = 'pod depth estimate'
    else:
        normalized = _local_depth(frame)
        source = 'local depth proxy'
    colorized = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_INFERNO)[..., ::-1]
    return colorized, source


def _occupancy(frame):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for _, _, (x1, y1, x2, y2) in boxes(frame):
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    mask = cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8))
    overlay = (frame.astype(np.float32) * 0.25).astype(np.uint8)
    occupied = mask > 0
    overlay[occupied] = (
        overlay[occupied] * 0.4 + np.array([0, 200, 80]) * 0.6
    ).astype(np.uint8)
    return overlay


def _detection(frame):
    overlay = frame.copy()
    known_names = objects.labels_by_class_id()
    detector_names = class_names()
    for class_id, confidence, (x1, y1, x2, y2) in boxes(frame):
        label = known_names.get(
            class_id,
            detector_names.get(class_id, f'class {class_id}'),
        )
        color = (60, 200, 90) if class_id in known_names else (240, 170, 40)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f'{label} {confidence:.2f}',
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _process_live(frame, mode):
    if mode == CHOICES[0]:
        return frame, ''
    if mode == CHOICES[1]:
        return _depth(frame)
    if mode == CHOICES[2]:
        return _occupancy(frame), ''
    if mode == CHOICES[3]:
        return gradient(frame), ''
    if mode == CHOICES[4]:
        return _detection(frame), ''
    raise ValueError(f'Unsupported camera view mode: {mode}')


def tick_all(mode, views, camera_status=''):
    main, front, corner, live = views
    if main is None:
        main = np.zeros((540, 960, 3), dtype=np.uint8)
    if front is None:
        front = np.zeros((540, 960, 3), dtype=np.uint8)
    if corner is None:
        corner = np.zeros((540, 960, 3), dtype=np.uint8)
    if live is None:
        live = np.zeros((480, 640, 3), dtype=np.uint8)

    error = ''
    view_note = ''
    try:
        live, view_note = _process_live(live, mode)
    except (cv2.error, OSError, RuntimeError, ValueError) as exception:
        LOGGER.exception('Could not render live camera mode %s', mode)
        error = f'Perception view unavailable: {exception}'

    details = (
        '### Simulation Details\n\n'
        'G1 simulation · cameras: main / front3m / corner  '
        f'· mode: {mode}'
    )
    if camera_status:
        details += f'\n\n{camera_status}'
    if view_note:
        details += f'\n\n{view_note}'
    if error:
        details += f'\n\n**{error}**'
    return main, front, corner, live, details
