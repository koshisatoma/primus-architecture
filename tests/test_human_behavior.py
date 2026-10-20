"""Focused tests for the Session 1–4 cognition prototypes."""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


class MemoryTests(unittest.TestCase):
    def setUp(self):
        from primus.memory_store import MemoryStore

        self.tempdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tempdir.name) / 'memory.sqlite3')

    def tearDown(self):
        self.tempdir.cleanup()

    def test_salience_order_and_exact_recall(self):
        self.store.store('episodic', {'utterance': 'likes coffee'}, salience=0.9)
        self.store.store('episodic', {'utterance': 'saw a cup'}, salience=0.3)
        recalled = self.store.recall('episodic', query='coffee', drift=0)
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0]['content']['utterance'], 'likes coffee')

    def test_decay_and_consolidation_are_persistent(self):
        memory_id = self.store.store('episodic', {'note': 'important'}, salience=0.9)
        before = self.store.recall('episodic', limit=1)[0]['salience']
        self.store.decay_all(now=time.time() + 3600)
        after = self.store.recall('episodic', limit=1)[0]['salience']
        self.assertLess(after, before)
        self.assertEqual(self.store.consolidate(0.7), 1)
        self.assertTrue(self.store.recall('episodic', limit=1)[0]['consolidated'])
        self.assertTrue(self.store.forget(memory_id))
        self.assertEqual(self.store.recall('episodic'), [])

    def test_rejects_non_json_finite_scores(self):
        with self.assertRaises(ValueError):
            self.store.store('episodic', {'value': float('nan')})

    def test_reconstructive_recall_is_non_destructive_by_default(self):
        from primus.reconstructive_recall import ReconstructiveRecall

        source = {'value': 3.0, 'nested': {'items': [1, 2]}}
        result = ReconstructiveRecall().recall_episode(source)
        self.assertEqual(source, result)
        self.assertIsNot(source, result)

    def test_skill_adapter_saves_and_loads_state(self):
        from primus.anti_forgetting import AntiForgetting

        with tempfile.TemporaryDirectory() as directory:
            anti_forgetting = AntiForgetting(directory)
            adapter = anti_forgetting.create_skill_adapter('grasp', 8, rank=2)
            anti_forgetting.save_skill_adapter('grasp')
            loaded = anti_forgetting.load_skill_adapter('grasp', 8, rank=2)
            self.assertIsNotNone(loaded)
            self.assertEqual(tuple(loaded.lora_A.shape), (8, 2))
            self.assertTrue(
                torch_equal(adapter.lora_A.detach(), loaded.lora_A.detach())
            )

    def test_ewc_penalty_has_finite_value(self):
        import torch
        from primus.anti_forgetting import AntiForgetting

        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            data = [
                {
                    'input': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    'target': torch.tensor([0, 1]),
                }
            ]
            ewc = AntiForgetting(directory)
            ewc.compute_fisher(model, data, 'task')
            self.assertTrue(torch.isfinite(ewc.ewc_penalty(model, 'task')).item())


def torch_equal(left, right):
    import torch

    return torch.equal(left, right)


class SocialCognitionTests(unittest.TestCase):
    def test_turn_taking_requires_multiple_people_and_speech(self):
        from primus.social_cognition import AudioTurnTaking

        now = time.monotonic()
        detector = AudioTurnTaking()
        detector.update(0.08, now)
        detector.update(0.08, now + 0.1)
        self.assertTrue(detector.humans_conversing(2))
        self.assertFalse(detector.humans_conversing(1))

    def test_attention_target_is_time_limited(self):
        from primus.social_cognition import TheoryOfMind

        attention = TheoryOfMind(timeout=1.0)
        self.assertEqual(
            attention.update_gaze(0.8, [('left', -0.5), ('right', 0.7)]),
            'right',
        )
        self.assertEqual(attention.get_primed_object(), 'right')
        attention.last_update -= 2
        self.assertIsNone(attention.get_primed_object())

    def test_prosody_returns_features_not_claimed_valence(self):
        from primus.social_cognition import EmotionRecognition

        output = EmotionRecognition().analyze_prosody(np.ones(1600) * 0.1)
        self.assertGreater(output['arousal'], 0)
        self.assertEqual(output['valence'], 0.0)

    def test_emotion_is_bounded_and_decays(self):
        from primus.affect_personality import EmotionSystem

        emotion = EmotionSystem()
        emotion.update_from_interaction(2.0, 2.0)
        self.assertEqual((emotion.valence, emotion.arousal), (1.0, 1.0))
        emotion.decay_towards_baseline(now=emotion.last_decay + 10)
        self.assertLess(emotion.valence, 1.0)

    def test_personality_rejects_unknown_trait(self):
        from primus.affect_personality import Personality

        with self.assertRaises(ValueError):
            Personality().update('X', 1.0)

    def test_proactive_prompt_requires_idle_threshold(self):
        from primus.proactive_agent import ProactiveInitiation, TimeSense

        time_sense = TimeSense()
        time_sense.last_interaction -= 3700
        proactive = ProactiveInitiation(time_sense)
        should_speak, text = proactive.should_initiate()
        self.assertTrue(should_speak)
        self.assertTrue(text)
        self.assertFalse(proactive.should_initiate()[0])

    def test_self_awareness_confidence_bands(self):
        from primus.proactive_agent import SelfAwareness

        awareness = SelfAwareness()
        awareness.update_confidence(0.2, 0.3)
        self.assertTrue(awareness.get_uncertainty_prefix())
        with self.assertRaises(ValueError):
            awareness.update_confidence(2.0, 0.0)


class SpatialAndIntegrationTests(unittest.TestCase):
    def test_spatial_memory_merges_and_returns_fixed_vector(self):
        from primus.spatial_memory import SpatialMemory

        memory = SpatialMemory()
        memory.update([('cup', [1.0, 0.5, 0.8], 41)], timestamp=10)
        memory.update([('cup', [1.05, 0.5, 0.8], 41)], timestamp=11)
        np.testing.assert_allclose(memory.query('cup'), [1.015, 0.5, 0.8])
        self.assertEqual(memory.get_state_vector().shape, (15,))
        self.assertEqual(len(memory.get_obstacles()), 1)

    def test_spatial_memory_validates_positions(self):
        from primus.spatial_memory import SpatialMemory

        with self.assertRaises(ValueError):
            SpatialMemory().update([('cup', [1, 2], 1)])

    def test_audio_localization_tracks_side(self):
        from primus.proprioception import AuditoryLocalization

        yaw, _ = AuditoryLocalization().update([0.1, 0.8])
        self.assertGreater(yaw, 0)

    def test_cognition_memory_hooks_and_context(self):
        from primus.integration import PrimusCognition
        from primus.memory_store import MemoryStore

        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / 'memory.sqlite3')
            cognition = PrimusCognition(store=store)
            context = cognition.think_hook('I like tea')
            self.assertIn('Conversation style', context['system_prompt'])
            cognition.respond_hook('I will remember that you said this.')
            self.assertEqual(
                store.recall('episodic', limit=1)[0]['content']['utterance'],
                'I like tea',
            )
            next_context = cognition.think_hook('What did I say?')
            self.assertIn('I like tea', next_context['system_prompt'])

    def test_action_parser_compiles_pick_and_place(self):
        from primus.skills_action import parse_action

        plan = parse_action('pick the cup and place it on the table')
        self.assertEqual(
            [item['prim'] for item in plan],
            ['reach', 'grasp', 'lift', 'walk_to', 'place'],
        )
        self.assertTrue(all(item['obj'] == 'cup' for item in plan))
        self.assertTrue(all(item['target'] == 'table' for item in plan))

    def test_fall_recovery_does_not_create_placeholder_command(self):
        import mujoco
        from primus.motor_planning import FallRecoveryStateMachine
        from primus.proprioception import FallDetector

        model = mujoco.MjModel.from_xml_path('sim/g1/scene_mjx.xml')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        detector = FallDetector(model, data, confirmation_seconds=0)
        self.assertEqual(detector.update(now=10), 'standing')
        data.xpos[detector.torso, 2] = 0.2
        recovery = FallRecoveryStateMachine(detector)
        with self.assertRaisesRegex(RuntimeError, 'no validated get-up policy'):
            recovery.get_override(np.zeros(29))

    def test_advisory_navigation_is_bounded(self):
        import mujoco
        from primus.motor_planning import ClutteredLocomotion
        from primus.spatial_memory import SpatialMemory

        model = mujoco.MjModel.from_xml_path('sim/g1/scene_mjx.xml')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        command = ClutteredLocomotion(
            SpatialMemory(), model, data
        ).compute_command(np.array([2.0, 0.5, 0.8]))
        self.assertEqual(command.shape, (3,))
        self.assertTrue(np.isfinite(command).all())
        self.assertLessEqual(np.max(np.abs(command)), 1.0)


if __name__ == '__main__':
    unittest.main()
