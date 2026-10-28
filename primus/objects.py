import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import cv2

from primus.detect import PERSON, boxes
from primus.people import most_recent_person


LOGGER = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
OBJECTS_FILE = DATA_DIR / 'objects.json'
NAME_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9 _-]{0,63}\Z')
MAX_PERSON_BOX_DIAGONAL_FRACTION = 0.05


def _load():
    if not OBJECTS_FILE.exists():
        return {}
    with OBJECTS_FILE.open(encoding='utf-8') as objects_file:
        objects = json.load(objects_file)
    if not isinstance(objects, dict):
        raise ValueError(f'{OBJECTS_FILE} must contain a JSON object')
    return objects


def _save(objects):
    OBJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=OBJECTS_FILE.parent,
            prefix='.objects-',
            suffix='.tmp',
            delete=False,
        ) as objects_file:
            temporary_path = Path(objects_file.name)
            json.dump(objects, objects_file, indent=2)
            objects_file.flush()
            os.fsync(objects_file.fileno())
        os.replace(temporary_path, OBJECTS_FILE)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _bbox_gap(person_bbox, object_bbox):
    px1, py1, px2, py2 = person_bbox
    ox1, oy1, ox2, oy2 = object_bbox
    horizontal_gap = max(px1 - ox2, ox1 - px2, 0)
    vertical_gap = max(py1 - oy2, oy1 - py2, 0)
    person_diagonal = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5
    if person_diagonal <= 0:
        return float('inf')
    return ((horizontal_gap**2 + vertical_gap**2) ** 0.5) / person_diagonal


def ground_object(name, frame, taught_by=None):
    """Remember a detected object whose box overlaps or nearly touches a person.

    Bounding-box proximity is a visual heuristic, not proof of physical contact.
    """
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name.strip()):
        raise ValueError('Object name must be 1-64 letters, numbers, spaces, _ or -')
    if frame is None:
        return False

    try:
        detections = boxes(frame)
    except cv2.error as error:
        LOGGER.exception('Could not detect an object near the person')
        raise RuntimeError('Object detector image processing failed') from error
    except (OSError, RuntimeError, ValueError):
        LOGGER.exception('Could not detect an object near the person')
        raise
    persons = [item for item in detections if item[0] == PERSON]
    objects = [item for item in detections if item[0] != PERSON]
    if not persons or not objects:
        return False

    candidates = [
        (_bbox_gap(persons[0][2], item[2]), item)
        for item in objects
    ]
    distance, closest = min(candidates, key=lambda item: item[0])
    if distance > MAX_PERSON_BOX_DIAGONAL_FRACTION:
        return False

    class_id, confidence, bbox = closest
    if taught_by is None:
        taught_by = most_recent_person()
    record = {
        'class_id': int(class_id),
        'confidence': float(confidence),
        'bbox': [int(value) for value in bbox],
        'taught_by': taught_by,
        'timestamp': time.time(),
        'grounding_method': 'person-object bounding-box proximity',
    }
    objects_by_name = _load()
    objects_by_name[name.strip().casefold()] = record
    try:
        _save(objects_by_name)
    except OSError:
        LOGGER.exception('Could not save grounded object %s', name)
        raise
    return True


def get_object(name):
    return _load().get(name.strip().casefold())


def list_objects():
    return list(_load())


def labels_by_class_id():
    """Return the most recently taught display name for each detector class."""
    labels = {}
    timestamps = {}
    for name, record in _load().items():
        if not isinstance(record, dict) or 'class_id' not in record:
            continue
        try:
            class_id = int(record['class_id'])
            timestamp = float(record.get('timestamp', 0))
        except (TypeError, ValueError):
            continue
        if timestamp >= timestamps.get(class_id, float('-inf')):
            labels[class_id] = name
            timestamps[class_id] = timestamp
    return labels
