import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BrainBridge(Node):
    """Relay voice text to the command topic and log executor status."""

    def __init__(self):
        super().__init__('primus_brain_bridge')
        self._command_publisher = self.create_publisher(
            String, '/primus/command', 10
        )
        self._voice_subscription = self.create_subscription(
            String, '/primus/voice_text', self._on_voice_text, 10
        )
        self._status_subscription = self.create_subscription(
            String, '/primus/status', self._on_status, 10
        )
        self.get_logger().info(
            'Brain bridge ready; listens on /primus/voice_text'
        )

    def _on_voice_text(self, message):
        command = String()
        command.data = message.data
        self._command_publisher.publish(command)

    def _on_status(self, message):
        self.get_logger().info(f'Executor status: {message.data}')


def main(args=None):
    rclpy.init(args=args)
    node = BrainBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
