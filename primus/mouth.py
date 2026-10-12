import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from primus.bus import put


VOICE = Path(__file__).resolve().parent.parent / 'data' / 'en_US-hfc_female-medium.onnx'


class Mouth:
    def __init__(self):
        self.v = None

    def _voice(self):
        if self.v is None and VOICE.exists():
            from piper import PiperVoice

            self.v = PiperVoice.load(VOICE)
        return self.v

    def say(self, text):
        voice = self._voice()
        if voice is None:
            print('[mouth]', text)
            return

        chunks = list(voice.synthesize(text[:300]))
        if not chunks:
            return
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
        started = time.perf_counter()
        sd.play(audio, chunks[0].sample_rate)
        put(ttfa_ms=round((time.perf_counter() - started) * 1000, 1))
        sd.wait()
