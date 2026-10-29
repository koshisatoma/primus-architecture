import json
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

# Updated paths for Known/Unknown split
KNOWN_FILE = DATA_DIR / 'detection' / 'known' / 'people.json'
UNKNOWN_FILE = DATA_DIR / 'detection' / 'unknown' / 'faces.json'
KNOWN_FACES_DIR = DATA_DIR / 'detection' / 'known' / 'faces'
KNOWN_EMBEDDINGS_DIR = DATA_DIR / 'detection' / 'known' / 'embeddings'

NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]{0,31}\Z")

_pending_names = []
_pending_face_indices = []
_pending_frame = None


def _ensure_dirs():
    """Ensure all detection directories exist."""
    KNOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    UNKNOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    KNOWN_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def _load_known():
    _ensure_dirs()
    if not KNOWN_FILE.exists():
        return {}
    with KNOWN_FILE.open(encoding='utf-8') as f:
        people = json.load(f)
    if not isinstance(people, dict):
        raise ValueError(f'{KNOWN_FILE} must contain a JSON object')
    return people


def _save_known(people):
    _ensure_dirs()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=KNOWN_FILE.parent,
            prefix='.people-', suffix='.tmp', delete=False,
        ) as f:
            temporary_path = Path(f.name)
            json.dump(people, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, KNOWN_FILE)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_unknown():
    _ensure_dirs()
    if not UNKNOWN_FILE.exists():
        return {}
    with UNKNOWN_FILE.open(encoding='utf-8') as f:
        return json.load(f)


def _save_unknown(unknown_faces):
    _ensure_dirs()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=UNKNOWN_FILE.parent,
            prefix='.unknown-', suffix='.tmp', delete=False,
        ) as f:
            temporary_path = Path(f.name)
            json.dump(unknown_faces, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, UNKNOWN_FILE)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def register_unknown_face(embedding, metadata=None):
    """
    Register a detected face that is not yet in the known database.
    Args:
        embedding: numpy array of DINOv2 embedding (or list)
        metadata: dict with optional info (e.g., {'location': 'kitchen', 'time': ...})
    Returns:
        str: The generated unknown_face_id
    """
    unknown = _load_unknown()
    face_id = str(uuid.uuid4())[:8]
    
    unknown[face_id] = {
        "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
        "first_seen": time.time(),
        "last_seen": time.time(),
        "metadata": metadata or {}
    }
    _save_unknown(unknown)
    LOGGER.info("Registered unknown face: %s", face_id)
    return face_id


def promote_unknown_to_known(unknown_id, name, embedding=None):
    """
    Move an unknown face to the known database with a given name.
    This is triggered when someone says "This is [Name]" or during spelling confirmation.
    """
    unknown = _load_unknown()
    known = _load_known()
    
    if unknown_id not in unknown:
        LOGGER.warning("Unknown face %s not found, creating new known entry anyway.", unknown_id)
        face_data = {"first_seen": time.time()}
        if embedding is not None:
            face_data["embedding"] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
    else:
        face_data = unknown.pop(unknown_id)
    
    # Add to known
    known[name] = {
        "face_image": f"detection/known/faces/{name}.png",
        "first_seen": face_data.get("first_seen", time.time()),
        "introductions": int(face_data.get("introductions", 0)) + 1,
        "metadata": face_data.get("metadata", {})
    }
    
    # Save embedding separately for fast numpy loading during recognition
    if "embedding" in face_data and face_data["embedding"] is not None:
        emb_path = KNOWN_EMBEDDINGS_DIR / f"{name}.npy"
        np.save(emb_path, np.array(face_data["embedding"]))
    
    _save_unknown(unknown)
    _save_known(known)
    LOGGER.info("Promoted %s to known person: %s", unknown_id, name)
    return True


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
        raise RuntimeError('No frontal-face cascade is installed.')
    
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f'Could not load face detector: {cascade_path}')
    
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    if face_index is None:
        return max(faces, key=lambda box: int(box[2]) * int(box[3]))
    
    faces = sorted(faces, key=lambda box: int(box[0]) + int(box[2]) // 2)
    if not 0 <= face_index < len(faces):
        return None
    return faces[face_index]


def capture_face(name, frame, face_index=None, embedding=None):
    """Save a detected face crop and its embedding after an explicit spoken self-introduction."""
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ValueError('Name must contain 1-32 letters, apostrophes, or hyphens')
    if frame is None:
        return False
    if getattr(frame, 'ndim', 0) != 3 or frame.shape[2] != 3:
        raise ValueError('Face capture requires an RGB image')

    _ensure_dirs()
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
        
        success, encoded = cv2.imencode('.png', cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
        if not success:
            raise OSError('OpenCV could not encode the detected face image')

        face_path = KNOWN_FACES_DIR / f'{name}.png'
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb', dir=KNOWN_FACES_DIR, prefix=f'.{name}-', suffix='.png.tmp', delete=False,
            ) as image_file:
                temporary_path = Path(image_file.name)
                image_file.write(encoded.tobytes())
                image_file.flush()
                os.fsync(image_file.fileno())
            os.replace(temporary_path, face_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

        # Save embedding if provided (e.g., from DINOv2 in svc_percept)
        if embedding is not None:
            emb_path = KNOWN_EMBEDDINGS_DIR / f"{name}.npy"
            np.save(emb_path, np.array(embedding))

        people = _load_known()
        previous = people.get(name, {})
        if not isinstance(previous, dict):
            raise ValueError(f'Existing record for {name!r} must be a JSON object')
            
        people[name] = {
            'face_image': str(face_path.relative_to(DATA_DIR)),
            'first_seen': previous.get('first_seen', time.time()),
            'introductions': int(previous.get('introductions', 0)) + 1,
        }
        _save_known(people)
        return True
        
    except cv2.error as error:
        LOGGER.exception('Could not capture face for %s', name)
        raise RuntimeError(f'Face detector failed for {name}') from error
    except (OSError, ValueError) as e:
        LOGGER.exception('Could not capture face for %s: %s', name, e)
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
    _pending_face_indices = list(range(len(unique_names))) if len(unique_names) > 1 else [None]
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


def confirm_spelling(spelling, embedding=None):
    """Confirm the next introduced name and save its corresponding face + embedding."""
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
    if not capture_face(name, _pending_frame, face_index=face_index, embedding=embedding):
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
    return list(_load_known())


def most_recent_person():
    people = _load_known()
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