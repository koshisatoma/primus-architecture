import logging
import threading

from primus.brain import think
from primus.ears import Ears
from primus.mouth import Mouth


LOGGER = logging.getLogger(__name__)
_stop = threading.Event()
_lock = threading.Lock()
_thread = None


def _loop():
    ears, mouth = Ears(), Mouth()
    from primus.integration import cognition
    from primus.camera_state import latest_teacher_frame
    from primus.people import has_pending_introduction
    from primus.social_awareness import SocialAwareness

    cognitive_state = cognition()
    cognitive_state.start_daemons()
    social_awareness = SocialAwareness()
    while not _stop.is_set():
        try:
            text = ears.listen(4)
            if text and social_awareness.should_respond(
                text,
                latest_teacher_frame(),
                interaction_active=has_pending_introduction(),
            ):
                print('[heard]', text)
                mouth.say(think(text))
        except Exception:
            LOGGER.exception('Voice agent iteration failed')
            _stop.wait(2)


def start():
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=_loop, daemon=True, name='primus-agent')
        _thread.start()


def stop():
    _stop.set()
    with _lock:
        thread = _thread
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=5)
    from primus.integration import cognition

    cognition().stop_daemons()
