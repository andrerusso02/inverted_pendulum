#!/usr/bin/env python3

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import rclpy
from control_msgs.msg import DynamicJointState
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from ros_pendulum import pendulum_inference_pipeline

JOINT_ACTUATED = 'gim6010'
JOINT_UNACTUATED = 'free_0'


CONTROL_FREQUENCY = 100.0  # Hz

POLICY_PATH = '/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-07-25_20-04-54/exported/policy.onnx'

STATIC_POSITION_OFFSET_RAD = 0.0


class JitterTracker:
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
    __slots__ = ['effort', 'position', 'velocity']

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
        
        self._inference_pipeline = pendulum_inference_pipeline.build_pendulum_inference_pipeline(
            policy_path=POLICY_PATH,
            log_function=self.get_logger().info,
            use_fake=False
        )

        self.timer = self.create_timer(1.0 / CONTROL_FREQUENCY, self.on_control_timer)
        self._jitter_tracker = JitterTracker(expected_period_ns=int(self.timer.timer_period_ns))

    def on_control_timer(self):
        self._jitter_tracker.log(self.get_logger().warn)
        
        # Wait until state is fully populated by the subscriber
        if self.robot_state.joint_actuated_values is None or self.robot_state.joint_unactuated_values is None:
            return

        if None in (
            self.robot_state.joint_actuated_values.position,
            self.robot_state.joint_actuated_values.velocity,
            self.robot_state.joint_unactuated_values.position,
        ):
            self.get_logger().warn("Received None values in joint states, skipping inference.")
            return

        outputs = self._inference_pipeline.execute({
            "gim_pos": self.robot_state.joint_actuated_values.position,
            "gim_vel": self.robot_state.joint_actuated_values.velocity,
            "free_pos": self.robot_state.joint_unactuated_values.position,
        })
        
        self.publish_command(outputs["gim_pos"])

    def publish_command(self, target_position: float):
        msg = Float64MultiArray()
        msg.data = [target_position]
        self.cmd_pub.publish(msg)

    def on_joint_state(self, msg: DynamicJointState):
        self.robot_state.update_from_joint_state(msg)

        
def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()