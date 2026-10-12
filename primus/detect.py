import logging
import threading
from pathlib import Path

import cv2


LOGGER = logging.getLogger(__name__)
_model = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_model_kind = None
_warmed_up = False
PERSON = 0
MAX_INFERENCE_WIDTH = 960
MAX_INFERENCE_HEIGHT = 540
MODEL_FILES = (
    ('yolo26x.engine', 'YOLO26x TensorRT'),
    ('yolo26x.pt', 'YOLO26x PyTorch'),
    ('yolo11x.engine', 'YOLO11x TensorRT'),
    ('yolo11x.pt', 'YOLO11x PyTorch'),
    ('yolo11n.engine', 'YOLO11n TensorRT'),
    ('yolo11n.pt', 'YOLO11n PyTorch'),
)


def _load():
    global _model, _model_kind
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO

                model_dir = Path(__file__).resolve().parents[1]
                failures = []
                for filename, kind in MODEL_FILES:
                    model_path = model_dir / filename
                    if not model_path.is_file():
                        continue
                    try:
                        _model = YOLO(str(model_path))
                        _model_kind = kind
                        LOGGER.info('Loaded detector: %s', model_path.name)
                        break
                    except Exception as error:
                        failures.append(f'{filename}: {error}')
                        LOGGER.warning(
                            'Could not load detector candidate %s',
                            filename,
                            exc_info=True,
                        )
                if _model is None:
                    detail = (
                        '; '.join(failures)
                        if failures
                        else 'no model files found'
                    )
                    raise RuntimeError(
                        f'No detection model could be loaded: {detail}'
                    )
    return _model


def _detections(result):
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        if class_id != PERSON and confidence <= 0.40:
            continue
        if class_id == PERSON and confidence <= 0.25:
            continue
        coordinates = [int(value) for value in box.xyxy[0]]
        detections.append((class_id, confidence, coordinates))
    detections.sort(key=lambda row: (row[0] != PERSON, -row[1]))
    return detections


def _inference_image(image):
    height, width = image.shape[:2]
    scale = min(
        1.0,
        MAX_INFERENCE_WIDTH / width,
        MAX_INFERENCE_HEIGHT / height,
    )
    if scale == 1.0:
        return image
    size = (round(width * scale), round(height * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def boxes(frame):
    """Return class, confidence, and xyxy detections, with people first."""
    if frame is None:
        return []
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    inference_frame = _inference_image(bgr_frame)
    with _inference_lock:
        result = _load()(inference_frame, verbose=False, conf=0.25, iou=0.45)[0]
    scale_x = frame.shape[1] / inference_frame.shape[1]
    scale_y = frame.shape[0] / inference_frame.shape[0]
    return [
        (
            class_id,
            confidence,
            [
                round(x1 * scale_x),
                round(y1 * scale_y),
                round(x2 * scale_x),
                round(y2 * scale_y),
            ],
        )
        for class_id, confidence, (x1, y1, x2, y2) in _detections(result)
    ]


def overlay(frame):
    global _warmed_up
    if frame is None:
        return None
    try:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        inference_frame = _inference_image(bgr_frame)
        with _inference_lock:
            result = _load()(inference_frame, verbose=False, conf=0.25, iou=0.45)[0]
        annotated_frame = inference_frame.copy()
        for class_id, confidence, (x1, y1, x2, y2) in _detections(result):
            color = (40, 220, 40) if class_id == PERSON else (255, 160, 40)
            label = f'{result.names[class_id]} {confidence:.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_top = max(y1, text_height + baseline + 2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(
                annotated_frame,
                (x1, label_top - text_height - baseline - 2),
                (x1 + text_width + 2, label_top),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 1, label_top - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        if annotated_frame.shape[:2] != frame.shape[:2]:
            annotated_frame = cv2.resize(
                annotated_frame,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    except Exception:
        LOGGER.exception('Object detector failed to process a webcam frame')
        raise


def warmup(frame):
    """Call once at startup to initialize TensorRT execution contexts."""
    global _warmed_up
    if _warmed_up:
        return
    LOGGER.info('Warming up YOLO (first inference is slow)...')
    overlay(frame)
    _warmed_up = True
    LOGGER.info('Warmup complete.')


def model_kind():
    return _model_kind