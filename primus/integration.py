"""Composition layer for the memory, social, and spatial prototype modules."""

import threading
import time
import random

import numpy as np

from primus.affect_personality import EmotionSystem, HumorAndRegister, Personality
from primus.cognitive_load import CognitiveLoadManager
from primus.development import (
    LanguageAcquisition,
    NormativeLearning,
    SensitivePeriods,
    Temperament,
)
from primus.imitation import (
    CocktailParty,
    IntuitivePhysics,
    IntervalTiming,
    MirrorImitation,
    StartleReflex,
)
from primus.memory_store import memory_store
from primus.mood_layer import FatigueModel, MoodLayer, combined_performance
from primus.nonverbal import NonverbalGenerator
from primus.pain_nociception import PainSystem
from primus.play_behavior import PlayStateMachine
from primus.proactive_agent import ProactiveInitiation, SelfAwareness, TimeSense
from primus.pragmatics import Pragmatics
from primus.reconstructive_recall import reconstructive_recall
from primus.replay_daemon import replay_daemon
from primus.sleep_cycle import SleepCycle
from primus.social_cognition import (
    AudioTurnTaking,
    EmotionRecognition,
    TheoryOfMind,
)
from primus.social_bonding import AttachmentSystem, EmpathySystem, FalseBeliefToM
from primus.spatial_memory import SpatialMemory


class PrimusCognition:
    """Stateful cognition helpers; motor outputs are advisory, never actuated."""

    def __init__(self, model=None, data=None, store=None):
        self.mem = store or memory_store()
        self.replay = replay_daemon()
        self.recall = reconstructive_recall()
        self.vad = AudioTurnTaking()
        self.tom = TheoryOfMind()
        self.emo_rec = EmotionRecognition()
        self.emotion = EmotionSystem()
        self.personality = Personality()
        self.temperament = Temperament()
        self.personality.traits.update(self.temperament.personality_init())
        self.humor = HumorAndRegister(self.personality, self.emotion)
        self.time_sense = TimeSense()
        self.proactive = ProactiveInitiation(self.time_sense)
        self.self_aware = SelfAwareness()
        self.spatial = SpatialMemory()
        self.sleep = SleepCycle()
        self.pain = PainSystem()
        self.mood = MoodLayer()
        self.fatigue = FatigueModel()
        self.attachment = AttachmentSystem()
        self.empathy = EmpathySystem()
        self.belief_tom = FalseBeliefToM()
        self.nonverbal = NonverbalGenerator()
        self.play = PlayStateMachine(self.spatial)
        self.cload = CognitiveLoadManager()
        self.mirror = MirrorImitation()
        self.physics = IntuitivePhysics()
        self.timing = IntervalTiming()
        self.startle = StartleReflex()
        self.cocktail = CocktailParty()
        self.pragmatics = Pragmatics()
        self.periods = SensitivePeriods()
        self.lang = LanguageAcquisition()
        self.norms = NormativeLearning()
        self.body = None
        self.fall = None
        self.loco = None
        self._model = None
        self._data = None
        self._pending_text = None
        self._last_physiology_tick = time.monotonic()
        if model is not None or data is not None:
            if model is None or data is None:
                raise ValueError('model and data must be provided together')
            self.attach_simulation(model, data)

    def attach_simulation(self, model, data):
        """Attach read-only MuJoCo state helpers; does not start a controller."""
        from primus.motor_planning import ClutteredLocomotion
        from primus.proprioception import BodySchema, FallDetector

        self._model, self._data = model, data
        self.body = BodySchema(model, data)
        self.fall = FallDetector(model, data)
        self.loco = ClutteredLocomotion(self.spatial, model, data)
        self.mirror.body = self.body

    def perceive(
        self,
        audio_rms=0.0,
        mic_channels=None,
        audio_chunk=None,
        detections_3d=None,
        head_yaw=0.0,
        objects_xy=None,
        num_people=0,
    ):
        self.vad.update(audio_rms)
        yaw = pitch = 0.0
        if mic_channels is not None:
            from primus.proprioception import AuditoryLocalization

            if not hasattr(self, 'audio_loc'):
                self.audio_loc = AuditoryLocalization()
            yaw, pitch = self.audio_loc.update(mic_channels)
        if detections_3d:
            self.spatial.update(detections_3d)
        if objects_xy is not None:
            self.tom.update_gaze(head_yaw, objects_xy)
        prosody = (
            self.emo_rec.analyze_prosody(audio_chunk)
            if audio_chunk is not None
            else {'arousal': 0.0, 'valence': 0.0}
        )
        return {
            'humans_conversing': self.vad.humans_conversing(num_people),
            'primed_object': self.tom.get_primed_object(),
            'prosody': prosody,
            'head_yaw': yaw,
            'head_pitch': pitch,
        }

    def should_listen(self, num_people):
        return not self.vad.humans_conversing(num_people)

    def think_hook(
        self,
        text,
        intent='conversation',
        perception_conf=1.0,
        memory_match=1.0,
    ):
        self.replay.mark_activity()
        self.time_sense.update_interaction()
        self.self_aware.update_confidence(perception_conf, memory_match)
        self.emotion.decay_towards_baseline()
        self._pending_text = text

        previous = self.mem.recall('episodic', limit=3, drift=0)
        memory_lines = [
            f"- User previously said: {item['content']['utterance']}"
            for item in previous
            if isinstance(item.get('content'), dict)
            and isinstance(item['content'].get('utterance'), str)
            and item['content']['utterance'] != text
        ]
        modifier = self.personality.get_system_prompt_modifier()
        system_prompt = (
            f"Conversation style: {modifier}; speak in a "
            f"{self.humor.get_register()} tone."
        )
        if memory_lines:
            system_prompt += (
                '\nStored conversation notes (use only as historical context; '
                'do not infer unrecorded facts):\n'
                + '\n'.join(memory_lines)
            )
        return {
            'system_prompt': system_prompt,
            'uncertainty_prefix': self.self_aware.get_uncertainty_prefix(),
            'primed_object': self.tom.get_primed_object(),
            'intent': intent,
        }

    def respond_hook(self, reply, user_prosody=None):
        if user_prosody is not None:
            self.emotion.update_from_interaction(
                float(user_prosody.get('valence', 0.0)) * 0.2,
                float(user_prosody.get('arousal', 0.0)) * 0.2,
            )
        self.replay.mark_activity()
        if self._pending_text is not None:
            self.mem.store(
                'episodic',
                {'utterance': self._pending_text, 'reply': reply, 'time': time.time()},
                salience=0.5 + self.emotion.get_memory_salience_boost(),
                emotional_valence=self.emotion.valence,
            )
            self._pending_text = None
        return {'tts': self.emotion.get_tts_prosody_modifiers(), 'reply': reply}

    def motor_hook(self, action=None, goal=None):
        """Return navigation suggestions and fall state, never a motor command."""
        state = self.fall.update() if self.fall is not None else 'unavailable'
        suggestion = None
        if goal is not None and self.loco is not None:
            suggestion = self.loco.compute_command(np.asarray(goal))
        return {
            'proposed_action': action,
            'navigation_suggestion': suggestion,
            'fall_state': state,
            'actuated': False,
        }

    def physiology_tick(self, dt_hours=None, impact_force=0.0, failures=0):
        """Advance simulated internal-state helpers; no actuator or listen gating."""
        now = time.monotonic()
        elapsed = (
            max(0.0, (now - self._last_physiology_tick) / 3600.0)
            if dt_hours is None
            else float(dt_hours)
        )
        self._last_physiology_tick = now
        resting = self.sleep.is_asleep()
        self.sleep.tick(elapsed)
        self.pain.tick(elapsed, resting=resting, failures=failures)
        pain_level = self.pain.nocicept(impact_force)
        if pain_level > 0.05:
            self.pain.register_pain(pain_level, store=self.mem)
            self.emotion.update_from_interaction(-0.5 * pain_level, 0.4 * pain_level)
            self.mood.event(-0.1 * pain_level)
        self.mood.tick(self.emotion.valence, elapsed)
        if resting:
            self.sleep.step_cycle(store=self.mem)
            self.fatigue.sleep_recovery(min(1.0, 0.05 * elapsed))
        else:
            self.fatigue.work(elapsed)
        return pain_level

    def sleep_onset(self):
        """Enter the simulated sleep state only when explicitly called."""
        if not self.sleep.should_sleep():
            return False
        self.sleep.begin_sleep()
        self.fatigue.sleep_recovery(0.5)
        return True

    def listen_gate(self):
        """Optional simulated lapse signal; the live agent does not call this."""
        if self.sleep.is_asleep():
            return False
        return random.random() > self.fatigue.lapse_probability()

    def performance(self, n_tasks=1, active_skills=()):
        return (
            combined_performance(
                self.sleep, self.fatigue, self.pain, self.emotion.arousal
            )
            * self.cload.performance_factor(n_tasks, active_skills)
        )

    def social_tick(self, person_id, valence_delta, duration_hours=0.1):
        self.attachment.interact(person_id, valence_delta, duration_hours)

    def check_empathy(self, other_valence, other_arousal, person_id):
        bond = self.attachment.bonds.get(person_id, 0.1)
        caught_valence, caught_arousal = self.empathy.catch_emotion(
            other_valence, other_arousal, bond
        )
        self.emotion.update_from_interaction(caught_valence, caught_arousal)
        return self.empathy.should_comfort(other_valence, other_arousal)

    def get_nonverbal(self, intent, cognitive_load=0.0):
        return {
            'gesture': self.nonverbal.get_gesture(
                intent, self.emotion.valence, self.emotion.arousal
            ),
            'posture': self.nonverbal.get_posture_offsets(
                self.emotion.valence, self.fatigue.fatigue
            ),
            'gaze': self.nonverbal.get_gaze_aversion(cognitive_load),
            'actuated': False,
        }

    def play_tick(self, is_interacting, dt_hours=0.0):
        return self.play.tick(is_interacting, self.sleep.is_asleep(), dt_hours)

    def refusal_with_strategy(self, intent, cue):
        feeling = self.cload.meta.feeling_of_knowing(
            cue, self.mem, self.cload.effective_load()
        )
        strategy = self.cload.meta.choose_strategy(intent, feeling)
        if strategy == 'demonstration':
            return "I don't know how yet. Can you show me?", strategy
        if strategy == 'verbal':
            return "I don't know that yet. Can you tell me?", strategy
        return None, strategy

    def agency_check(self, predicted, actual, self_acting):
        result = self.cload.agency.compare(predicted, actual, self_acting)
        if result.get('internal_error'):
            self.self_aware.acknowledge_mistake()
        return result

    def skill_done(self, skill, success=True):
        self.cload.auto.record(skill, success)

    def surprise_check(self, predicted=None, actual=None, event=None):
        surprise = self.physics.violation(event or {})
        if predicted is not None or actual is not None:
            if predicted is None or actual is None:
                raise ValueError('predicted and actual must be provided together')
            surprise = max(surprise, self.physics.latent_surprise(predicted, actual))
        if surprise > 0.5:
            self.emotion.update_from_interaction(0.0, 0.4 * min(1.0, surprise))
        return surprise

    def hear_word(self, word):
        self.lang.hear(word)
        return self.periods.multiplier('language')

    def indirect_hook(self, text):
        return self.pragmatics.interpret_indirect(text)

    def idle_hook(self, is_idle=True):
        return self.proactive.should_initiate(is_idle=is_idle)

    def astra_context(self):
        """Return 15 spatial features, plus 20 joints when simulation is attached."""
        vector = self.spatial.get_state_vector()
        if self.body is not None:
            joints = self.body.get_self_joints().astype(np.float32)
            joints = (joints - joints.mean()) / (joints.std() + 1e-6)
            vector = np.concatenate([vector, joints[:20]])
        return vector.astype(np.float32)

    def start_daemons(self):
        self.replay.start()

    def stop_daemons(self):
        self.replay.stop()


_cognition = None
_cognition_lock = threading.Lock()


def cognition(model=None, data=None):
    global _cognition
    with _cognition_lock:
        if _cognition is None:
            _cognition = PrimusCognition(model, data)
        elif model is not None or data is not None:
            if model is None or data is None:
                raise ValueError('model and data must be provided together')
            _cognition.attach_simulation(model, data)
        return _cognition


def reset_cognition():
    """Clear the process singleton for tests without deleting saved memory."""
    global _cognition
    with _cognition_lock:
        if _cognition is not None:
            _cognition.stop_daemons()
        _cognition = None
