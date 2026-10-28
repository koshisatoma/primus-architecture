import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from primus import viewops
from primus.cloud_client import depth_cloud


class VisionViewTests(unittest.TestCase):
    def test_gradient_and_local_depth_preserve_frame_dimensions(self):
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[:, 24:] = 255

        edge_view = viewops.gradient(frame)
        depth_view = viewops._local_depth(frame)

        self.assertEqual(edge_view.shape, frame.shape)
        self.assertEqual(edge_view.dtype, np.uint8)
        self.assertEqual(depth_view.shape, frame.shape[:2])
        self.assertEqual(depth_view.dtype, np.uint8)
        self.assertGreater(int(edge_view.max()), 0)

    def test_occupancy_highlights_detected_boxes(self):
        frame = np.full((24, 24, 3), 100, dtype=np.uint8)
        detection = (1, 0.9, (8, 8, 16, 16))
        with patch.object(viewops, 'boxes', return_value=[detection]):
            output = viewops._occupancy(frame)

        self.assertEqual(output.shape, frame.shape)
        self.assertGreater(int(output[12, 12, 1]), int(output[12, 12, 0]))

    def test_depth_uses_local_proxy_without_waiting_for_cloud(self):
        frame = np.full((24, 32, 3), 120, dtype=np.uint8)
        with (
            patch.object(viewops, '_DEPTH_FUTURE', None),
            patch.object(viewops, '_DEPTH_RESULT', None),
            patch.object(viewops, '_DEPTH_REQUEST_AFTER', 10**12),
        ):
            output, source = viewops._depth(frame)

        self.assertEqual(output.shape, frame.shape)
        self.assertEqual(source, 'local depth proxy')

    def test_detection_uses_taught_class_names(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with (
            patch.object(viewops.objects, 'labels_by_class_id', return_value={1: 'cup'}),
            patch.object(viewops, 'class_names', return_value={1: 'bottle'}),
            patch.object(
                viewops,
                'boxes',
                return_value=[(1, 0.95, (2, 2, 20, 20))],
            ),
        ):
            output = viewops._detection(frame)

        self.assertEqual(output.shape, frame.shape)
        self.assertGreater(int(output.sum()), 0)

    def test_rgb_mode_and_empty_camera_views_return_expected_outputs(self):
        live = np.full((8, 12, 3), 50, dtype=np.uint8)
        frames = (None, None, None, live)

        main, front, corner, output, details = viewops.tick_all(
            viewops.CHOICES[0], frames, 'LIVE · KITCHEN'
        )

        self.assertEqual(main.shape, (540, 960, 3))
        self.assertEqual(front.shape, main.shape)
        self.assertEqual(corner.shape, main.shape)
        np.testing.assert_array_equal(output, live)
        self.assertIn('LIVE · KITCHEN', details)

    def test_cloud_depth_decodes_little_endian_dimensions_from_response(self):
        response = Mock()
        response.headers = {'X-Depth-Width': '3', 'X-Depth-Height': '2'}
        response.content = np.array([1, 2, 3, 4, 5, 6], dtype='<u2').tobytes()
        with (
            patch.dict(os.environ, {'PRIMUS_POD': 'http://localhost'}),
            patch('primus.cloud_client.requests.post', return_value=response) as post,
        ):
            depth = depth_cloud(b'jpeg-data')

        self.assertEqual(depth.shape, (2, 3))
        np.testing.assert_array_equal(depth, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(post.call_args.args[0], 'http://localhost:8002/depth')


if __name__ == '__main__':
    unittest.main()
