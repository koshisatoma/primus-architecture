import json
import os
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def _load_primus_modules():
    configured_root = os.environ.get('PRIMUS3_ROOT')
    if configured_root:
        repository_root = Path(configured_root).resolve()
    else:
        repository_root = next(
            (
                parent
                for parent in Path(__file__).resolve().parents
                if (parent / 'primus' / 'detect.py').is_file()
            ),
            None,
        )
        if repository_root is None:
            raise RuntimeError(
                'Could not locate primus/detect.py. Set PRIMUS3_ROOT to the '
                'Primus project directory.'
            )
    if not (repository_root / 'primus' / 'detect.py').is_file():
        raise RuntimeError(
            'Could not find primus/detect.py. Set PRIMUS3_ROOT to the '
            'Primus project directory.'
        )
    sys.path.insert(0, str(repository_root))
    from primus.detect import boxes
    from primus.shmcam import ShmCam

    return ShmCam, boxes


class PerceptionBridge(Node):
    def __init__(self):
        super().__init__('primus_perception_bridge')
        ShmCam, self._detect_boxes = _load_primus_modules()
        self._camera = ShmCam()
        self._image_publisher = self.create_publisher(
            Image, '/primus/camera/image', 10
        )
        self._detection_publisher = self.create_publisher(
            String, '/primus/detections', 10
        )
        self._timer = self.create_timer(0.1, self._publish_frame)
        self.get_logger().info('Perception bridge ready at 10 Hz')

    def _publish_frame(self):
        frame = self._camera.get()
        if frame is None:
            return

        image = Image()
        image.height, image.width = frame.shape[:2]
        image.encoding = 'rgb8'
        image.is_bigendian = False
        image.step = image.width * 3
        image.data = frame.tobytes()
        self._image_publisher.publish(image)

        detections = [
            {'class': class_id, 'confidence': confidence, 'bbox': bbox}
            for class_id, confidence, bbox in self._detect_boxes(frame)
        ]
        message = String()
        message.data = json.dumps(detections)
        self._detection_publisher.publish(message)

    def destroy_node(self):
        if getattr(self, '_camera', None) is not None:
            self._camera.close()
            self._camera = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
