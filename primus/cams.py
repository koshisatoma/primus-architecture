import logging
import threading
import time

from primus.detect import model_kind, overlay
from primus.bus import put
from primus.camera_state import publish_teacher_frame
from primus.render import Sim
from primus.webcam import Webcam


LOGGER = logging.getLogger(__name__)
NAMED = ['kitchen', 'room', 'g1_eye', 'g1_body', 'teacher', 'detect']
FEEDS = NAMED
DETECTION_INTERVAL = 1 / 10


class Cams:
    def __init__(self):
        try:
            from primus.shmcam import ShmCam

            self.web = ShmCam()
            self._using_shm_camera = True
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.warning(
                'C++ shared-memory camera unavailable; using OpenCV fallback: %s',
                error,
            )
            self.web = Webcam()
            self._using_shm_camera = False
        self.frames = {name: None for name in FEEDS}
        self._focus = 'kitchen'
        self._orbit = (200.0, -15.0, 3.5)
        self.fps = 0.0
        self.render_error = None
        self.detection_error = None
        self.detection_ready = False
        self.detection_fps = 0.0
        self._frame_sequence = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name='primus-render'
        )
        self._detect_thread = threading.Thread(
            target=self._detect_loop, daemon=True, name='primus-detect'
        )
        self._render_thread.start()
        self._detect_thread.start()

    @property
    def focus(self):
        with self._lock:
            return self._focus

    @focus.setter
    def focus(self, name):
        if name not in FEEDS:
            raise ValueError(f'Unknown camera feed: {name}')
        with self._lock:
            self._focus = name
    @property
    def orbit(self):
        with self._lock:
            return self._orbit

    @orbit.setter
    def orbit(self, values):
        azimuth, elevation, distance = values
        with self._lock:
            self._orbit = (float(azimuth), float(elevation), float(distance))

    def _render_loop(self):
        sim = None
        try:
            import mujoco

            sim = Sim()
            torso_id = mujoco.mj_name2id(
                sim.m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link'
            )
            if torso_id < 0:
                torso_id = mujoco.mj_name2id(
                    sim.m, mujoco.mjtObj.mjOBJ_BODY, 'torso'
                )
            last, count = time.monotonic(), 0
            secondary_cameras = ['room', 'g1_eye', 'g1_body']
            secondary_index = 0
            while not self._stop.is_set():
                started = time.monotonic()
                if torso_id >= 0:
                    sim.d.xfrc_applied[torso_id, 2] = 80.0
                mujoco.mj_step(sim.m, sim.d)
                rendered = {'kitchen': sim.orbit(*self.orbit)}
                camera = secondary_cameras[secondary_index]
                rendered[camera] = sim.named(camera)
                secondary_index = (secondary_index + 1) % len(secondary_cameras)
                with self._lock:
                    self.frames.update(rendered)
                    self.frames['teacher'] = self.web.get()
                    publish_teacher_frame(self.frames['teacher'])
                    if self._frame_sequence == 0 and self.frames['teacher'] is not None:
                        from primus.detect import warmup
                        warmup(self.frames['teacher'])
                    if self._using_shm_camera and self._frame_sequence % 30 == 0:
                        _, timestamp_ns, capture_fps = self.web.stats()
                        put(
                            cam_fps=capture_fps,
                            cam_lag_ms=round(
                                (time.time_ns() - timestamp_ns) / 1e6, 1
                            ),
                            cam_res=(
                                f'{self.web.width}x{self.web.height}'
                            ),
                        )
                    self._frame_sequence += 1
                    count += 1
                    elapsed = time.monotonic() - last
                    if elapsed >= 1.0:
                        self.fps = count / elapsed
                        count, last = 0, time.monotonic()
                        put(fps=self.fps)
                self._stop.wait(max(0.0, 1 / 30 - (time.monotonic() - started)))
        except Exception as error:
            LOGGER.exception('MuJoCo camera rendering stopped')
            with self._lock:
                self.render_error = str(error)
        finally:
            self.web.close()
            if sim is not None:
                try:
                    sim.close()
                except Exception:
                    LOGGER.exception('Could not close the MuJoCo renderer')

    def close(self):
        self._stop.set()
        publish_teacher_frame(None)
        self.web.close()
        self._render_thread.join(timeout=5)
        self._detect_thread.join(timeout=5)

    def _detect_loop(self):
        sequence = -1
        last_rate_update = time.monotonic()
        last_inference = 0.0
        inference_count = 0
        while not self._stop.is_set():
            with self._lock:
                current_sequence = self._frame_sequence
                teacher_frame = self.frames.get('teacher')
            if teacher_frame is None or current_sequence == sequence:
                self._stop.wait(0.005)
                continue
            delay = DETECTION_INTERVAL - (time.monotonic() - last_inference)
            if delay > 0:
                self._stop.wait(delay)
                continue
            try:
                started = time.perf_counter()
                detected = overlay(teacher_frame)
                put(yolo_ms=round((time.perf_counter() - started) * 1000, 1))
            except Exception as error:
                LOGGER.exception('YOLO segmentation overlay stopped')
                with self._lock:
                    self.detection_error = str(error)
                    return
            with self._lock:
                self.frames['detect'] = detected
                sequence = current_sequence
                self.detection_ready = True
                last_inference = time.monotonic()
                inference_count += 1
                elapsed = time.monotonic() - last_rate_update
                if elapsed >= 1.0:
                    self.detection_fps = inference_count / elapsed
                    inference_count = 0
                    last_rate_update = time.monotonic()

    def view(self):
        with self._lock:
            frame = self.frames.get(self._focus)
            if self._focus == 'detect' and frame is None:
                frame = self.frames.get('teacher')
            return frame

    def names(self):
        return list(FEEDS)

    def status(self):
        with self._lock:
            focus = self._focus.upper()
            is_detection_view = self._focus == 'detect'
            is_teacher_view = self._focus == 'teacher'
            fps = self.fps
            render_error = self.render_error
            detection_error = self.detection_error
            detection_ready = self.detection_ready
            detection_fps = self.detection_fps
        if render_error:
            return f'RENDER ERROR · {render_error}'
        if is_detection_view:
            if detection_error:
                return f'DETECTION ERROR · {detection_error}'
            kind = model_kind() or 'detection'
            state = f'{kind.upper()} LIVE' if detection_ready else f'LOADING {kind.upper()}'
            return (
                f'LIVE · {focus} · {state} · {detection_fps:.1f} detect fps'
                f' · {fps:.1f} sim fps'
            )
        if is_teacher_view:
            return f'{self.web.status.upper()} · {focus} · {fps:.1f} sim fps'
        return f'LIVE · {focus} · {fps:.1f} sim fps'
