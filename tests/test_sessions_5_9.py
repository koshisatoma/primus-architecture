"""Regression tests for the bounded Sessions 5–9 cognition helpers."""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


class PhysiologyTests(unittest.TestCase):
    def test_sleep_cycle_consolidates_prunes_and_recombines_memories(self):
        from primus.memory_store import MemoryStore
        from primus.sleep_cycle import SleepCycle

        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / 'memory.sqlite3')
            store.store('episodic', {'event': 'one'}, salience=0.9)
            store.store('episodic', {'event': 'two'}, salience=0.5)
            store.store('episodic', {'event': 'low'}, salience=0.1)
            cycle = SleepCycle()
            cycle.tick(12)
            self.assertTrue(cycle.should_sleep())
            self.assertTrue(cycle.begin_sleep())
            self.assertFalse(cycle.wake(0.3))
            nrem = cycle.step_cycle(store)
            self.assertGreaterEqual(nrem['consolidated'], 1)
            self.assertEqual(nrem['pruned'], 1)
            rem = cycle.step_cycle(store)
            self.assertEqual(len(rem['dreams']), 1)
            self.assertIn('dream', rem['dreams'][0])
            self.assertTrue(cycle.wake(1.0))
            self.assertLess(cycle.performance_multiplier(), 1.0)

    def test_pain_is_bounded_and_vectors_are_advisory_values(self):
        from primus.pain_nociception import PainSystem

        pain = PainSystem()
        self.assertEqual(pain.nocicept(40), 0.0)
        self.assertGreater(pain.nocicept(120), 0.3)
        pain.register_pain(0.8, cue_key='surface')
        withdrawal = pain.protective_reflex([0, 0, 0.8], [0, 0.2, 0.9])
        self.assertIsNotNone(withdrawal)
        self.assertGreater(withdrawal[1], 0)
        avoidance = pain.avoidance_vector(
            [0, 0.8, 0.8], {'surface': np.array([0, 0.5, 0.8])}
        )
        self.assertGreater(np.linalg.norm(avoidance), 0)
        self.assertLessEqual(np.linalg.norm(avoidance), 1.0)
        with self.assertRaises(ValueError):
            pain.nocicept(-1)

    def test_mood_fatigue_and_performance_bounds(self):
        from primus.mood_layer import FatigueModel, MoodLayer
        from primus.pain_nociception import PainSystem
        from primus.sleep_cycle import SleepCycle
        from primus.mood_layer import combined_performance

        mood = MoodLayer()
        self.assertGreater(mood.tick(0.8, dt_hours=6), 0.3)
        mood.event(-1.0)
        self.assertLess(mood.mood, 0.3)
        fatigue = FatigueModel()
        fatigue.work(dt_hours=3)
        self.assertGreater(fatigue.fatigue, 0.2)
        self.assertGreater(fatigue.lapse_probability(), 0)
        fatigue.break_rest(dt_hours=1.0)
        self.assertGreater(fatigue.vigilance, 0.5)
        value = combined_performance(SleepCycle(), fatigue, PainSystem(), 0.5)
        self.assertGreater(value, 0)
        self.assertLessEqual(value, 1)


class SocialDevelopmentTests(unittest.TestCase):
    def test_bond_empathy_and_false_belief_helpers(self):
        from primus.social_bonding import (
            AttachmentSystem,
            EmpathySystem,
            FalseBeliefToM,
        )

        attachment = AttachmentSystem()
        for _ in range(20):
            attachment.interact('person-a', 0.8, 1.0)
        attachment.interact('person-b', -0.2, 0.1)
        self.assertEqual(attachment.primary_caregiver, 'person-a')
        self.assertLess(attachment.get_proxemics_preference('person-a'), 1.0)
        emotion = EmpathySystem()
        caught, _ = emotion.catch_emotion(-0.8, 0.5, 0.8)
        self.assertLess(caught, -0.2)
        self.assertTrue(emotion.should_comfort(-0.6, 0.5)[0])
        tom = FalseBeliefToM()
        tom.teacher_sees('cup', [1, 0, 0.8])
        tom.update_reality('cup', [1, 0, 0.8])
        self.assertFalse(tom.has_false_belief('cup'))
        tom.update_reality('cup', [2, 0, 0.8])
        self.assertTrue(tom.has_false_belief('cup'))
        self.assertEqual(tom.get_correction_strategy('cup'), 'explain_and_point')

    def test_nonverbal_and_play_are_suggestions_only(self):
        from primus.nonverbal import NonverbalGenerator
        from primus.play_behavior import PlayStateMachine
        from primus.spatial_memory import SpatialMemory

        generator = NonverbalGenerator()
        self.assertEqual(generator.get_gesture('greeting', 0.2, 0.5), 'wave')
        posture = generator.get_posture_offsets(-0.8, 0.8)
        self.assertLess(posture['head_pitch'], -0.1)
        spatial = SpatialMemory()
        spatial.update([('toy', np.array([1.0, 1.0, 0.5]), 1)])
        play = PlayStateMachine(spatial)
        play._initiate_play()
        result = play.tick(False, False, 0.001)
        self.assertIsNotNone(result)
        self.assertFalse(result['actuated'])


class CognitiveAndDevelopmentTests(unittest.TestCase):
    def test_working_memory_agency_metacognition_and_automaticity(self):
        from primus.cognitive_load import (
            CognitiveLoadManager,
            Metacognition,
            SenseOfAgency,
            SkillAutomaticity,
            WorkingMemory,
        )

        working = WorkingMemory()
        for index in range(6):
            working.hold(index, salience=index / 10)
        self.assertEqual(len(working.slots), 4)
        self.assertEqual(working.slots[0]['content'], 2)
        self.assertLess(working.dual_task_cost(2), 1.0)
        agency = SenseOfAgency(threshold=0.3)
        self.assertTrue(agency.compare(np.zeros(2), np.ones(2) * 0.01, True)['agency'])
        self.assertTrue(
            agency.compare(np.zeros(2), np.ones(2), True)['internal_error']
        )
        self.assertTrue(
            agency.compare(np.zeros(2), np.ones(2), False)['external_event']
        )
        meta = Metacognition()
        self.assertLess(
            meta.judgment_of_learning(0.8, 0.9),
            meta.judgment_of_learning(0.8, 0.0),
        )
        automaticity = SkillAutomaticity()
        for _ in range(30):
            automaticity.record('walk')
        self.assertEqual(automaticity.stage('walk'), 'autonomous')
        manager = CognitiveLoadManager()
        self.assertLessEqual(manager.performance_factor(2), 1.0)

    def test_imitation_physics_timing_startle_and_attention(self):
        from primus.imitation import (
            CocktailParty,
            IntuitivePhysics,
            IntervalTiming,
            MirrorImitation,
            StartleReflex,
        )

        imitation = MirrorImitation()
        imitation.observe('wave', np.ones((3, 4)))
        np.testing.assert_array_equal(imitation.reproduce('wave'), np.ones((3, 4)))
        self.assertEqual(imitation.novelty('unknown'), 1.0)
        physics = IntuitivePhysics()
        self.assertGreater(physics.violation({'type': 'float'}), 0.5)
        self.assertGreater(physics.latent_surprise(np.zeros(2), np.ones(2)), 0.5)
        timing = IntervalTiming()
        timing.start(arousal=0.9)
        timing._start -= 0.05
        self.assertGreaterEqual(timing.estimate(), 0.05)
        startle = StartleReflex()
        startle.response(0.9)
        high_response = startle.response(0.5)
        first_weak = startle.response(0.2)
        last_weak = first_weak
        for _ in range(5):
            last_weak = startle.response(0.2)
        self.assertGreater(high_response, 0.5)
        self.assertLess(last_weak, first_weak)
        attention = CocktailParty(teacher_pitch=0.5)
        self.assertEqual(attention.update([(0.9, 0.4)], True), 'teacher')
        self.assertEqual(
            attention.update([(0.5, 0.4), (0.9, 0.4)]), 'teacher'
        )

    def test_pragmatics_and_developmental_helpers(self):
        from primus.development import (
            LanguageAcquisition,
            NormativeLearning,
            SensitivePeriods,
            Temperament,
        )
        from primus.pragmatics import Pragmatics

        pragmatic = Pragmatics()
        self.assertEqual(
            pragmatic.interpret_indirect("it's cold in here"), 'close_window'
        )
        self.assertIsNone(pragmatic.interpret_indirect('is it cold?'))
        self.assertIn('blue', pragmatic.repair('red', 'blue'))
        self.assertEqual(pragmatic.register(0.9, 0.5), 'intimate-casual')
        language = LanguageAcquisition()
        for word in ('cup', 'ball', 'mama', 'papa', 'milk', 'dog', 'cat', 'go', 'eat', 'more'):
            language.hear(word)
        self.assertEqual(language.stage(), 'one-word')
        self.assertIn('E', Temperament(42).personality_init())
        self.assertLessEqual(SensitivePeriods().multiplier('language'), 1.0)
        norms = NormativeLearning()
        norms.correction('share', -0.5)
        norms.correction('share', -0.5)
        self.assertIsNotNone(norms.violation_reaction('share'))


class IntegrationTests(unittest.TestCase):
    def test_facade_exposes_helpers_without_actuating(self):
        from primus.integration import PrimusCognition
        from primus.memory_store import MemoryStore

        with tempfile.TemporaryDirectory() as directory:
            cognition = PrimusCognition(
                store=MemoryStore(Path(directory) / 'memory.sqlite3')
            )
            cognition.social_tick('person-a', 0.5)
            self.assertEqual(cognition.get_nonverbal('greeting')['gesture'], 'wave')
            self.assertLessEqual(cognition.performance(), 1.0)
            self.assertGreater(cognition.surprise_check(event={'type': 'float'}), 0.5)
            self.assertEqual(cognition.indirect_hook("it's dark in here"), 'turn_on_light')
            self.assertGreater(cognition.physiology_tick(dt_hours=0.1, impact_force=120), 0)
            result = cognition.motor_hook()
            self.assertFalse(result['actuated'])


if __name__ == '__main__':
    unittest.main()
