import logging
import os
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests


LOGGER = logging.getLogger(__name__)
POD_ENV = 'PRIMUS_POD'


def _url(port, path):
    base = os.environ.get(POD_ENV, '').strip()
    if not base:
        return None
    try:
        parsed = urlsplit(base)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            raise ValueError
        hostname = parsed.hostname
        if ':' in hostname:
            hostname = f'[{hostname}]'
        return urlunsplit(
            (parsed.scheme, f'{hostname}:{port}', path, '', '')
        )
    except ValueError:
        LOGGER.warning('Ignoring invalid %s URL', POD_ENV)
        return None


def chat_cloud(
    text,
    timeout=8,
    *,
    system_prompt=None,
    history=(),
):
    endpoint = _url(8000, '/v1/chat/completions')
    if endpoint is None:
        return None

    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.extend(history)
    messages.append({'role': 'user', 'content': text})
    try:
        response = requests.post(
            endpoint,
            json={
                'model': 'qwen-vl',
                'messages': messages,
                'max_tokens': 120,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Cloud chat returned an empty or invalid reply')
        return content.strip()
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        LOGGER.warning('Cloud chat unavailable; using local chat: %s', error)
        return None


def stt_cloud(pcm16_bytes, timeout=6):
    endpoint = _url(8002, '/transcribe')
    if endpoint is None:
        return None
    try:
        response = requests.post(
            endpoint,
            data=pcm16_bytes,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=timeout,
        )
        response.raise_for_status()
        text = response.json()['text']
        if not isinstance(text, str):
            raise ValueError('Cloud transcription response must contain text')
        return text.strip() or None
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        LOGGER.warning('Cloud transcription unavailable; using local Whisper: %s', error)
        return None


def depth_cloud(jpg_bytes, timeout=4):
    endpoint = _url(8002, '/depth')
    if endpoint is None:
        return None
    try:
        response = requests.post(
            endpoint,
            data=jpg_bytes,
            headers={'Content-Type': 'image/jpeg'},
            timeout=timeout,
        )
        response.raise_for_status()
        width = int(response.headers['X-Depth-Width'])
        height = int(response.headers['X-Depth-Height'])
        depth = np.frombuffer(response.content, dtype='<u2')
        if width <= 0 or height <= 0 or depth.size != width * height:
            raise ValueError('Cloud depth response has invalid dimensions')
        return depth.reshape(height, width)
    except (
        requests.RequestException,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        LOGGER.warning('Cloud depth unavailable; using local visualization: %s', error)
        return None
