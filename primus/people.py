import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import cv2


LOGGER = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
PEOPLE_FILE = DATA_DIR / 'people.json'
FACES_DIR = DATA_DIR / 'faces'
NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]{0,31}\Z")
_pending_names = []
_pending_face_indices = []
_pending_frame = None


def _load():
    if not PEOPLE_FILE.exists():
        return {}
    with PEOPLE_FILE.open(encoding='utf-8') as people_file:
        people = json.load(people_file)
    if not isinstance(people, dict):
        raise ValueError(f'{PEOPLE_FILE} must contain a JSON object')
    return people


def _save(people):
    PEOPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=PEOPLE_FILE.parent,
            prefix='.people-',
            suffix='.tmp',
            delete=False,
        ) as people_file:
            temporary_path = Path(people_file.name)
            json.dump(people, people_file, indent=2)
            people_file.flush()
            os.fsync(people_file.fileno())
        os.replace(temporary_path, PEOPLE_FILE)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _face_box(frame, face_index=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cascade_name = 'haarcascade_frontalface_default.xml'
    cascade_paths = (
        Path(cv2.data.haarcascades) / cascade_name,
        Path('/usr/share/opencv4/haarcascades') / cascade_name,
        Path('/usr/share/opencv/haarcascades') / cascade_name,
    )
    cascade_path = next((path for path in cascade_paths if path.is_file()), None)
    if cascade_path is None:
        raise RuntimeError(
            'No frontal-face cascade is installed; checked '
            + ', '.join(str(path) for path in cascade_paths)
        )
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f'Could not load face detector: {cascade_path}')
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None
    if face_index is None:
        return max(faces, key=lambda box: int(box[2]) * int(box[3]))
    faces = sorted(faces, key=lambda box: int(box[0]) + int(box[2]) // 2)
    if not 0 <= face_index < len(faces):
        return None
    return faces[face_index]


def capture_face(name, frame, face_index=None):
    """Save a detected face crop after an explicit spoken self-introduction."""
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ValueError('Name must contain 1-32 letters, apostrophes, or hyphens')
    if frame is None:
        return False
    if getattr(frame, 'ndim', 0) != 3 or frame.shape[2] != 3:
        raise ValueError('Face capture requires an RGB image')

    try:
        face = _face_box(frame, face_index=face_index)
        if face is None:
            LOGGER.info('No face detected while recording introduction for %s', name)
            return False

        x, y, width, height = (int(value) for value in face)
        pad_x, pad_y = round(width * 0.2), round(height * 0.2)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2 = min(frame.shape[1], x + width + pad_x)
        y2 = min(frame.shape[0], y + height + pad_y)
        face_rgb = frame[y1:y2, x1:x2]
        success, encoded = cv2.imencode(
            '.png', cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        )
        if not success:
            raise OSError('OpenCV could not encode the detected face image')

        FACES_DIR.mkdir(parents=True, exist_ok=True)
        face_path = FACES_DIR / f'{name}.png'
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb',
                dir=FACES_DIR,
                prefix=f'.{name}-',
                suffix='.png.tmp',
                delete=False,
            ) as image_file:
                temporary_path = Path(image_file.name)
                image_file.write(encoded.tobytes())
                image_file.flush()
                os.fsync(image_file.fileno())
            os.replace(temporary_path, face_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

        people = _load()
        previous = people.get(name, {})
        if not isinstance(previous, dict):
            raise ValueError(f'Existing record for {name!r} must be a JSON object')
        people[name] = {
            'face_image': str(face_path.relative_to(DATA_DIR.parent)),
            'first_seen': previous.get('first_seen', time.time()),
            'introductions': int(previous.get('introductions', 0)) + 1,
        }
        _save(people)
        return True
    except cv2.error as error:
        LOGGER.exception('Could not capture face for %s', name)
        raise RuntimeError(f'Face detector failed for {name}') from error
    except (OSError, ValueError):
        LOGGER.exception('Could not capture face for %s', name)
        raise


def start_introduction(names, frame):
    """Queue one or more names for spoken spelling confirmation."""
    global _pending_face_indices, _pending_frame, _pending_names
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError('An introduction must include at least one name')

    unique_names = []
    seen = set()
    for name in names:
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            raise ValueError('Each name must contain 1-32 letters, apostrophes, or hyphens')
        if name.casefold() not in seen:
            unique_names.append(name)
            seen.add(name.casefold())
    if not unique_names:
        raise ValueError('An introduction must include at least one name')

    _pending_names = unique_names
    _pending_face_indices = (
        list(range(len(unique_names))) if len(unique_names) > 1 else [None]
    )
    _pending_frame = frame.copy() if frame is not None else None
    if len(unique_names) == 1:
        return f'Nice to meet you, {unique_names[0]}! How do you spell that?'
    if len(unique_names) == 2:
        introductions = ' and '.join(unique_names)
    else:
        introductions = ', '.join(unique_names[:-1]) + f', and {unique_names[-1]}'
    return (
        f'Nice to meet you, {introductions}! '
        'I match names to faces from left to right. If I heard them out of order, '
        'please introduce everyone again from left to right. Then spell each name.'
    )


def has_pending_introduction():
    return bool(_pending_names)


def cancel_introduction():
    global _pending_face_indices, _pending_frame, _pending_names
    _pending_names = []
    _pending_face_indices = []
    _pending_frame = None


def confirm_spelling(spelling):
    """Confirm the next introduced name and save its corresponding face."""
    global _pending_face_indices, _pending_frame, _pending_names
    if not _pending_names:
        return "I wasn't expecting a spelling. Introduce yourself first."
    if not isinstance(spelling, str):
        raise ValueError('Spelling must be text')

    spelled_name = re.sub(r'[^a-z]', '', spelling.casefold())
    name = _pending_names[0]
    if spelled_name != name.casefold():
        return f"I heard {spelling!r}, but that doesn't match {name}. Try spelling it again?"

    face_index = _pending_face_indices[0]
    if not capture_face(name, _pending_frame, face_index=face_index):
        cancel_introduction()
        return (
            f"I couldn't save a face for {name}. Please introduce yourself again "
            'with your face visible.'
        )

    _pending_names.pop(0)
    _pending_face_indices.pop(0)
    if _pending_names:
        return f'Got it, {name} is saved! Now spell {_pending_names[0]} for me.'

    _pending_frame = None
    return f"Perfect, {name} is saved. I'll remember you!"


def list_people():
    return list(_load())


def most_recent_person():
    people = _load()
    if not people:
        return None
    return max(
        people,
        key=lambda name: float(
            people[name].get('first_seen', 0)
            if isinstance(people[name], dict)
            else 0
        ),
    )
