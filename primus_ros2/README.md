# Primus ROS 2 skeleton

For repository-wide installation, models, and setup instructions, see
[`../setup.md`](../setup.md). The project overview is in
[`../README.md`](../README.md).

This ROS 2 Humble package provides three topic bridges:

- `perception_bridge` publishes the shared-memory teacher camera as
  `/primus/camera/image` and YOLO detections as `/primus/detections`.
- `brain_bridge` forwards `/primus/voice_text` to `/primus/command` and logs
  `/primus/status`.
- `task_executor` classifies action commands and reports that the corresponding
  motion backend is unavailable. It does not move the robot.

Build from the repository root:

```bash
cd primus_ros2
source /opt/ros/humble/setup.bash
colcon build --packages-select primus_actions
source install/setup.bash
```

Run the nodes in separate terminals after starting the camera producer:

```bash
ros2 run primus_actions perception_bridge
ros2 run primus_actions task_executor
ros2 run primus_actions brain_bridge
```

The perception node needs the Primus Python dependencies installed in its
Python environment. When running an installed node, set `PRIMUS3_ROOT` to this
repository so the node can import `primus.detect` and `primus.shmcam`.

To exercise command routing without speech:

```bash
ros2 topic pub --once /primus/voice_text std_msgs/msg/String "{data: 'find the cup'}"
```

The social-awareness gate requires a wake word when two or more people are
visible. It cannot determine whether people are conversing; face-box counts are
only a conservative proxy for whether to ask for explicit attention.

Navigation, locomotion, sitting, grasping, and placing are not implemented.
The executor reports this explicitly rather than claiming an action succeeded.
