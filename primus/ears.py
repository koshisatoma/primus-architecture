import time

import sounddevice as sd
from faster_whisper import WhisperModel

from primus.bus import put


class Ears:
    INITIAL_PROMPT = (
        'Primus, Koshi, G1, MuJoCo, camera, kitchen, cup, mug, bowl, '
        'bottle, apple, banana, pan, knife.'
    )

    def __init__(self, sr=16000):
        self.sr = sr
        self.m = None

    def _model(self):
        if self.m is None:
            self.m = WhisperModel(
                'base',
                device='cpu',
                compute_type='int8',
                local_files_only=False,
            )
        return self.m

    def listen(self, seconds=4):
        import numpy as np

        model = self._model()
        buf = sd.rec(
            int(seconds * self.sr),
            samplerate=self.sr,
            channels=1,
            dtype='float32',
        )
        sd.wait()
        started = time.perf_counter()
        audio = buf.flatten()
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if 0 < peak < 0.1:
            audio = audio * (0.3 / peak)
        segments, _ = model.transcribe(
            audio,
            language='en',
            beam_size=5,
            best_of=3,
            temperature=0.0,
            initial_prompt=self.INITIAL_PROMPT,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={'min_silence_duration_ms': 300},
        )
        transcript = ' '.join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()
        put(stt_ms=round((time.perf_counter() - started) * 1000, 1))
        return transcript
