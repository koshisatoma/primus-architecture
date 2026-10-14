import os
import unittest
from unittest.mock import patch

import numpy as np

from primus import brain, cloud_client, people, social
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
    @patch('primus.ears.stt_cloud', return_value='cloud transcript')
    @patch('primus.ears.put')
    @patch('primus.ears.sd.wait')
    @patch('primus.ears.sd.rec')
    def test_cloud_transcription_skips_loading_local_whisper(
        self, record, wait, put, stt_cloud
    ):
        record.return_value = np.array([[0.05], [-0.025]], dtype=np.float32)

        with patch.object(Ears, '_model') as model_loader:
            transcript = Ears().listen(seconds=1)

        self.assertEqual(transcript, 'cloud transcript')
        stt_cloud.assert_called_once()
        self.assertEqual(
            put.call_args.kwargs['stt_src'],
            'cloud',
        )
        model_loader.assert_not_called()
        wait.assert_called_once()

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


class CloudChatTests(unittest.TestCase):
    def setUp(self):
        social.reset_conversation()

    def tearDown(self):
        social.reset_conversation()

    @patch('primus.social.requests.post')
    @patch('primus.social.chat_cloud', return_value='cloud reply')
    def test_cloud_chat_is_first_and_keeps_conversation(
        self, cloud_chat, local_request
    ):
        self.assertEqual(brain.chat('Hello'), 'cloud reply')
        self.assertEqual(brain.chat('How are you?'), 'cloud reply')

        self.assertEqual(cloud_chat.call_count, 2)
        self.assertEqual(
            cloud_chat.call_args.kwargs['history'],
            (
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'cloud reply'},
            ),
        )
        local_request.assert_not_called()

    @patch('primus.social.requests.post')
    @patch('primus.social.chat_cloud', return_value=None)
    def test_local_ollama_remains_fallback(self, cloud_chat, local_request):
        response = local_request.return_value
        response.json.return_value = {'message': {'content': 'local reply'}}

        self.assertEqual(brain.chat('Hello'), 'local reply')
        cloud_chat.assert_called_once()
        local_request.assert_called_once()


class CloudClientTests(unittest.TestCase):
    @patch.dict(os.environ, {'PRIMUS_POD': 'http://pod.example:8000'})
    @patch('primus.cloud_client.requests.post')
    def test_chat_targets_cloud_brain_and_sends_context(self, post):
        post.return_value.json.return_value = {
            'choices': [{'message': {'content': 'cloud reply'}}]
        }

        result = cloud_client.chat_cloud(
            'hello',
            system_prompt='persona',
            history=({'role': 'assistant', 'content': 'previous reply'},),
        )

        self.assertEqual(result, 'cloud reply')
        self.assertEqual(
            post.call_args.args[0],
            'http://pod.example:8000/v1/chat/completions',
        )
        self.assertEqual(
            post.call_args.kwargs['json']['messages'],
            [
                {'role': 'system', 'content': 'persona'},
                {'role': 'assistant', 'content': 'previous reply'},
                {'role': 'user', 'content': 'hello'},
            ],
        )

    @patch.dict(os.environ, {'PRIMUS_POD': 'http://pod.example:8000'})
    @patch('primus.cloud_client.requests.post')
    def test_transcription_targets_perception_service(self, post):
        post.return_value.json.return_value = {'text': 'cloud words'}

        self.assertEqual(cloud_client.stt_cloud(b'\x01\x00'), 'cloud words')
        self.assertEqual(post.call_args.args[0], 'http://pod.example:8002/transcribe')
        self.assertEqual(post.call_args.kwargs['data'], b'\x01\x00')

    @patch.dict(os.environ, {}, clear=True)
    @patch('primus.cloud_client.requests.post')
    def test_no_pod_configured_skips_remote_request(self, post):
        self.assertIsNone(cloud_client.chat_cloud('hello'))
        post.assert_not_called()


if __name__ == '__main__':
    unittest.main()
