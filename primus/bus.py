import fcntl
import json
import os
import tempfile
import time
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
METRICS_PATH = DATA_DIR / 'metrics.json'
LOCK_PATH = DATA_DIR / 'metrics.lock'


def put(**values):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open('a', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if METRICS_PATH.exists():
            with METRICS_PATH.open(encoding='utf-8') as metrics_file:
                current = json.load(metrics_file)
            if not isinstance(current, dict):
                raise ValueError(f'{METRICS_PATH} must contain a JSON object')
        else:
            current = {}

        current.update(values)
        current['t'] = time.time()
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=DATA_DIR,
                prefix='.metrics-',
                suffix='.tmp',
                delete=False,
            ) as metrics_file:
                temporary_path = Path(metrics_file.name)
                json.dump(current, metrics_file)
                metrics_file.flush()
            os.replace(temporary_path, METRICS_PATH)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def get():
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open(encoding='utf-8') as metrics_file:
        metrics = json.load(metrics_file)
    if not isinstance(metrics, dict):
        raise ValueError(f'{METRICS_PATH} must contain a JSON object')
    return metrics
