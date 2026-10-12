import re

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TaskExecutor(Node):
    """Classify requested actions and report unavailable motion backends."""

    def __init__(self):
        super().__init__('primus_task_executor')
        self._command_subscription = self.create_subscription(
            String, '/primus/command', self._on_command, 10
        )
        self._status_publisher = self.create_publisher(
            String, '/primus/status', 10
        )
        self.get_logger().info('Task executor ready; motion backends are not configured')

    def _on_command(self, message):
        command = message.data.strip()
        normalized = command.casefold()
        if re.search(r'\b(?:stand|sit|walk|move|navigate|go)\b', normalized):
            action = 'locomotion'
        elif re.search(r'\b(?:pick|grab|grasp|place|put)\b', normalized):
            action = 'manipulation'
        elif re.search(r'\b(?:find|locate)\b', normalized):
            action = 'navigation'
        else:
            self._publish_status(f"Unsupported command: {command}")
            return

        self._publish_status(
            f"Received {action} request '{command}', but no {action} "
            'backend is configured; no robot motion was executed.'
        )

    def _publish_status(self, text):
        message = String()
        message.data = text
        self._status_publisher.publish(message)
        self.get_logger().warning(text)


def main(args=None):
    rclpy.init(args=args)
    node = TaskExecutor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
