import time
import numpy as np
import onnxruntime as ort
from enum import StrEnum
from dataclasses import dataclass
import rclpy
from typing import Any
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import DynamicJointState

JOINT_ACTUATED = 'gim6010'
JOINT_UNACTUATED = 'free_0'


CONTROL_FREQUENCY = 50.0  # Hz

POLICY_PATH = '/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-07-22_20-32-27/exported/policy.onnx'

STATIC_POSITION_OFFSET_RAD = 0.0


class JitterTracker:
    """Tracks timer jitter by measuring real inter-callback intervals."""

    def __init__(self, expected_period_ns: int, threshold_ns: int = 1_000_000) -> None:
        self._expected_period_ns = expected_period_ns
        self._threshold_ns = threshold_ns
        self._last_ns: int | None = None

    def log(self, log_function) -> None:
        now = time.monotonic_ns()
        if self._last_ns is not None:
            actual_period_ns = now - self._last_ns
            jitter_ns = actual_period_ns - self._expected_period_ns
            if abs(jitter_ns) > self._threshold_ns:
                log_function(
                    f"Timer expected period: {self._expected_period_ns} ns, "
                    f"actual period: {actual_period_ns} ns, jitter: {jitter_ns} ns"
                )
        self._last_ns = now

class InterfaceName(StrEnum):
    POSITION = 'position'
    VELOCITY = 'velocity'
    EFFORT = 'effort'

class InterfaceValues:
    __slots__ = ['position', 'velocity', 'effort']

    def __init__(self, position: float | None = None, velocity: float | None = None, effort: float | None = None) -> None:
        self.position = position
        self.velocity = velocity
        self.effort = effort
        
    def __str__(self) -> str:
        return f"InterfaceValues(position={self.position}, velocity={self.velocity}, effort={self.effort})"

@dataclass
class RobotState:
    joint_actuated_values: InterfaceValues | None = None
    joint_unactuated_values: InterfaceValues | None = None

    def update_from_joint_state(self, joint_state_msg: DynamicJointState) -> None:
        for idx, joint_name in enumerate(joint_state_msg.joint_names):
            if joint_name == JOINT_ACTUATED:
                self.joint_actuated_values = self._get_interface_values(joint_state_msg.interface_values[idx])
            elif joint_name == JOINT_UNACTUATED:
                self.joint_unactuated_values = self._get_interface_values(joint_state_msg.interface_values[idx])

    @staticmethod
    def _get_interface_values(interface_values_entry: Any) -> InterfaceValues:
        val_map = dict(zip(interface_values_entry.interface_names, interface_values_entry.values))
        
        return InterfaceValues(
            position=val_map.get(InterfaceName.POSITION.value, None),
            velocity=val_map.get(InterfaceName.VELOCITY.value, None),
            effort=val_map.get(InterfaceName.EFFORT.value, None)
        )
    
    def __str__(self) -> str:
        return f"RobotState(\njoint_actuated_values=\n{self.joint_actuated_values},\njoint_unactuated_values=\n{self.joint_unactuated_values})"


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        self.robot_state = RobotState()
        
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, 
            '/joint_group_position_controller/commands', 
            10
        )
        self.state_sub = self.create_subscription(
            DynamicJointState, 
            '/dynamic_joint_states', 
            self.on_joint_state, 
            10
        )

        # Initialize ONNX session
        self.ort_session = ort.InferenceSession(POLICY_PATH)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        # Policy scaling parameters (Update these to match your Isaac Lab config)
        self.action_scale = 1.0 
        self.default_joint_pos = 0.0

        self.timer = self.create_timer(1.0 / CONTROL_FREQUENCY, self.on_control_timer)
        self._jitter_tracker = JitterTracker(expected_period_ns=int(self.timer.timer_period_ns))

    def on_control_timer(self):
        self._jitter_tracker.log(self.get_logger().warn)
        
        # Wait until state is fully populated by the subscriber
        if self.robot_state.joint_actuated_values is None or self.robot_state.joint_unactuated_values is None:
            return

        pos_gim = self.robot_state.joint_actuated_values.position
        vel_gim = self.robot_state.joint_actuated_values.velocity
        pos_free = self.robot_state.joint_unactuated_values.position

        # Ensure values are not None before inference
        if None in (pos_gim, vel_gim, pos_free):
            self.get_logger().warn("Received None values in joint states, skipping inference.")
            return

        # Construct observation vector. 
        # WARNING: The order must perfectly match the observation group in your Isaac Lab task.
        obs_array = np.array([[pos_gim, vel_gim, pos_free]], dtype=np.float32)

        # Run ONNX inference
        ort_outs = self.ort_session.run([self.output_name], {self.input_name: obs_array})
        
        # Extract raw action output
        raw_action = float(ort_outs[0][0][0])

        # Apply standard Isaac Lab scaling (target = action * scale + default_pos)
        target_position = (raw_action * self.action_scale) + self.default_joint_pos
        
        self.publish_command(target_position)

    def publish_command(self, target_position: float):
        msg = Float64MultiArray()
        msg.data = [target_position]
        self.cmd_pub.publish(msg)

    def on_joint_state(self, msg: DynamicJointState):
        self.robot_state.update_from_joint_state(msg)
        # self.get_logger().info(f"{self.robot_state}")

        
def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()