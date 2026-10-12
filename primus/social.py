import logging
import re
import threading

import requests


LOGGER = logging.getLogger(__name__)
OLLAMA_URL = 'http://127.0.0.1:11434/api/chat'
MODEL = 'qwen2.5:3b-instruct'
MAX_HISTORY_MESSAGES = 12
PERSONA = """You are Primus, a friendly robot chatting casually with someone you know.
Sound warm, relaxed, and natural, like an easy conversation—not a help desk or
an assistant giving a demo. Answer the point they actually made. Use contractions
and everyday language. Keep most replies to one short sentence; use a second only
when it adds something useful. Don't end every reply with a question, offer help,
or steer the conversation toward chores. Don't repeatedly mention being a robot,
your kitchen, or your limitations. Never invent observations, memories, feelings,
or personal experiences. Use the recent conversation for context, but only claim
to remember what is actually present in that conversation. If asked directly,
be straightforward that you are an AI robot. If asked for instructions for a
physical task that nobody has taught you, say you don't know and invite them to
show you. Do not use lists or markdown."""
_conversation = []
_conversation_lock = threading.Lock()
_GREETING = re.compile(r"^\s*(?:hi|hey|hello|good morning|good afternoon|good evening)\b", re.I)
_DISTRESS = re.compile(
    r"\b(?:rough day|bad day|stressed|upset|frustrated|exhausted|"
    r"feeling down|feeling low|had a hard time)\b",
    re.I,
)


def _fallback(text):
    if _GREETING.search(text):
        return "Hey, good to hear from you."
    if _DISTRESS.search(text):
        return "Ah, that sounds like a lot. Want to tell me what happened?"
    return "I lost my local chat connection for a moment—could you say that again?"


def chat(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Chat input must be a non-empty string')

    with _conversation_lock:
        messages = [
            {'role': 'system', 'content': PERSONA},
            *_conversation,
            {'role': 'user', 'content': text.strip()},
        ]
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    'model': MODEL,
                    'messages': messages,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()['message']['content'].strip()
            if not content:
                raise ValueError('Ollama returned an empty reply')
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            LOGGER.warning('Local Ollama chat unavailable: %s', error)
            return _fallback(text)

        _conversation.extend(
            [
                {'role': 'user', 'content': text.strip()},
                {'role': 'assistant', 'content': content},
            ]
        )
        del _conversation[:-MAX_HISTORY_MESSAGES]
        return content


def reset_conversation():
    """Clear recent chat context without deleting taught skills or names."""
    with _conversation_lock:
        _conversation.clear()
