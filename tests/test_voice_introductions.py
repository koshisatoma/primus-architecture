import unittest
from unittest.mock import patch

import numpy as np

from primus import brain, people
from primus.ears import Ears
from primus.social_awareness import SocialAwareness


class IntroductionFlowTests(unittest.TestCase):
    def setUp(self):
        people.cancel_introduction()
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def tearDown(self):
        people.cancel_introduction()

    @patch('primus.camera_state.latest_teacher_frame')
    @patch('primus.people.capture_face', return_value=True)
    def test_single_name_requires_spelling_confirmation(
        self, capture_face, latest_teacher_frame
    ):
        latest_teacher_frame.return_value = self.frame

        response = brain.think("Hey, I'm Koshi")
        self.assertEqual(
            response, 'Nice to meet you, Koshi! How do you spell that?'
        )
        self.assertTrue(people.has_pending_introduction())

        response = brain.think('K O S H I')
        self.assertEqual(
            response, "Perfect, Koshi is saved. I'll remember you!"
        )
        self.assertFalse(people.has_pending_introduction())
        capture_face.assert_called_once()
        self.assertEqual(capture_face.call_args.args[0], 'Koshi')
        self.assertIsNone(capture_face.call_args.kwargs['face_index'])

    @patch('primus.camera_state.latest_teacher_frame')
    @patch('primus.people.capture_face', return_value=True)
    def test_multi_name_flow_maps_faces_left_to_right(
        self, capture_face, latest_teacher_frame
    ):
        latest_teacher_frame.return_value = self.frame

        response = brain.think('This is Rahul and Priya')
        self.assertEqual(
            response,
            'Nice to meet you, Rahul and Priya! '
            'I match names to faces from left to right. If I heard them out of '
            'order, please introduce everyone again from left to right. Then spell '
            'each name.',
        )
        self.assertEqual(
            brain.think('R A H U L'),
            'Got it, Rahul is saved! Now spell Priya for me.',
        )
        self.assertEqual(
            brain.think('Priya'), "Perfect, Priya is saved. I'll remember you!"
        )
        self.assertEqual(
            [call.args[0] for call in capture_face.call_args_list],
            ['Rahul', 'Priya'],
        )
        self.assertEqual(
            [call.kwargs['face_index'] for call in capture_face.call_args_list],
            [0, 1],
        )

    @patch('primus.camera_state.latest_teacher_frame')
    @patch('primus.people.capture_face', return_value=True)
    def test_incorrect_spelling_keeps_current_name_pending(
        self, capture_face, latest_teacher_frame
    ):
        latest_teacher_frame.return_value = self.frame
        brain.think("I'm Rahul")

        self.assertIn("doesn't match Rahul", brain.think('Rohan'))
        self.assertTrue(people.has_pending_introduction())
        self.assertEqual(
            brain.think('Rahul'), "Perfect, Rahul is saved. I'll remember you!"
        )
        capture_face.assert_called_once()

    @patch('primus.camera_state.latest_teacher_frame')
    @patch('primus.people.capture_face', return_value=True)
    def test_mixed_multi_person_introduction(
        self, capture_face, latest_teacher_frame
    ):
        latest_teacher_frame.return_value = self.frame

        response = brain.think("I'm Koshi and this is Rahul")

        self.assertIn('Koshi and Rahul', response)
        self.assertTrue(people.has_pending_introduction())
        self.assertEqual(
            brain.think('K O S H I'),
            'Got it, Koshi is saved! Now spell Rahul for me.',
        )
        self.assertEqual(
            brain.think('R A H U L'),
            "Perfect, Rahul is saved. I'll remember you!",
        )
        self.assertEqual(
            [call.kwargs['face_index'] for call in capture_face.call_args_list],
            [0, 1],
        )

    @patch('primus.camera_state.latest_teacher_frame')
    @patch('primus.objects.ground_object', return_value=True)
    def test_object_introduction_without_article_stays_object_capture(
        self, ground_object, latest_teacher_frame
    ):
        latest_teacher_frame.return_value = self.frame

        response = brain.think('This is a cup')

        self.assertIn("Got it, that's a cup", response)
        ground_object.assert_called_once()
        self.assertFalse(people.has_pending_introduction())


class WhisperConfigurationTests(unittest.TestCase):
    @patch('primus.ears.put')
    @patch('primus.ears.sd.wait')
    @patch('primus.ears.sd.rec')
    def test_quiet_audio_is_boosted_and_transcribed_deterministically(
        self, record, wait, put
    ):
        audio = np.array([[0.05], [-0.025]], dtype=np.float32)
        record.return_value = audio

        class Model:
            def transcribe(self, samples, **kwargs):
                self.samples = samples
                self.kwargs = kwargs
                return [type('Segment', (), {'text': ' namaste '})()], None

        model = Model()
        with patch.object(Ears, '_model', return_value=model):
            transcript = Ears().listen(seconds=1)

        self.assertEqual(transcript, 'namaste')
        self.assertAlmostEqual(float(np.max(np.abs(model.samples))), 0.3)
        self.assertEqual(model.kwargs['language'], 'en')
        self.assertEqual(model.kwargs['beam_size'], 5)
        self.assertEqual(model.kwargs['best_of'], 3)
        self.assertEqual(model.kwargs['temperature'], 0.0)
        self.assertEqual(
            model.kwargs['vad_parameters'], {'min_silence_duration_ms': 300}
        )
        wait.assert_called_once()
        put.assert_called_once()


class SocialAwarenessTests(unittest.TestCase):
    @patch('primus.social_awareness.boxes')
    def test_requires_wake_word_when_multiple_people_are_visible(self, boxes):
        boxes.return_value = [
            (0, 0.9, [0, 0, 40, 80]),
            (0, 0.8, [50, 0, 90, 80]),
        ]
        awareness = SocialAwareness()

        self.assertFalse(awareness.should_respond('hello', self_frame(), False))
        self.assertTrue(awareness.should_respond('Hey, Primus!', None, False))
        self.assertTrue(awareness.should_respond('hello', None, True))
        boxes.assert_called_once()

    @patch('primus.social_awareness.boxes')
    def test_single_person_does_not_require_wake_word(self, boxes):
        boxes.return_value = [(0, 0.9, [0, 0, 40, 80])]

        self.assertTrue(SocialAwareness().should_respond('hello', self_frame()))


def self_frame():
    return np.zeros((1, 1, 3), dtype=np.uint8)


class VoiceActionRoutingTests(unittest.TestCase):
    @patch('primus.brain._send_ros_command')
    def test_actionable_voice_command_is_forwarded_to_ros(self, send_command):
        send_command.return_value = "Okay, find the cup. I'm on it."

        result = brain.think('Find the cup')

        self.assertEqual(result, "Okay, find the cup. I'm on it.")
        send_command.assert_called_once_with('Find the cup')


if __name__ == '__main__':
    unittest.main()
