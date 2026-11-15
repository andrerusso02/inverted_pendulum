#!/usr/bin/env python3

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import rclpy
from control_msgs.msg import DynamicJointState
from controller_manager_msgs.srv import ListControllers
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from ros_pendulum.pendulum_inference_pipeline import (
    PendulumPipelineInput,
    build_pendulum_inference_pipeline,
)

JOINT_ACTUATED = 'gim6010'
JOINT_UNACTUATED = 'free_0'
POSITION_CONTROLLER_NAME = 'joint_group_position_controller'


CONTROL_FREQUENCY = 100.0  # Hz

# If either joint state is older than this, stop inferring and hold the motor where it is.
# The 11-08 bag showed why: the encoder UDP stream died mid-swing and the node kept driving
# the motor into the -2 rad clip for 4 s using frozen pole data.
STATE_STALE_TIMEOUT_S = 0.1

# POLICY_PATH = '/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-07-26_10-17-42/exported/policy.onnx'
# POLICY_PATH = "/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-11-01_01-19-22/exported/policy.onnx"
# POLICY_PATH = "/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-11-08_16-31-07/exported/policy.onnx"
# POLICY_PATH = "/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-11-13_23-07-57/exported/policy.onnx"
# POLICY_PATH = "/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-11-13_23-46-59/exported/policy.onnx"
# POLICY_PATH = "/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-11-13_23-46-59/exported/policy.onnx"
# POLICY_PATH = "/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-11-14_09-29-43/exported/policy.onnx"
# POLICY_PATH = "/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-11-14_22-22-50/exported/policy.onnx"
# Iter 11 (23-03-29) REVERTED after 30 s hardware session: push-trained policy is higher-gain
# (hesitant start, rough hold: quiet tilt 12 deg vs iter-10's 0.1-5 deg, tau 1.18 vs 0.35 Nm)
# and the +direction failure persisted across both policies while sim push-probes are
# symmetric -> the asymmetry is hardware-side, not policy-side. iter-10 remains best on HW.
POLICY_PATH = "/home/andre/git/inverted_pendulum/Pendulum/logs/rsl_rl/cartpole_direct/2026-11-14_22-22-50/exported/policy.onnx"

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
        
        self._inference_pipeline = build_pendulum_inference_pipeline(
            policy_path=POLICY_PATH,
            log_function=self.get_logger().info,
            use_fake=False
        )

        self.timer = self.create_timer(1.0 / CONTROL_FREQUENCY, self.on_control_timer)
        self._jitter_tracker = JitterTracker(expected_period_ns=int(self.timer.timer_period_ns))

        # The ODrive only enters closed loop when joint_group_position_controller activates
        # (odrive_hw_interface set_axis_command_mode on interface claim). The 11-14 bag showed
        # inference starting ~0.5 s before that: the EMA filter slewed its command to the -2 clip
        # against an idle drive, which then executed the full step at once and flung the pole.
        # Gate inference on the controller being active; until then publish a hold-at-position
        # command so activation cannot slew the motor from a stale default setpoint either.
        self._drive_ready = False
        self._list_controllers_cli = self.create_client(
            ListControllers, '/controller_manager/list_controllers'
        )
        self._controller_check_timer = self.create_timer(0.2, self._check_controller_active)

        # freshness tracking + pipeline seeding (monotonic seconds, None = never received)
        self._gim_stamp: float | None = None
        self._free_stamp: float | None = None
        self._seeded = False
        self._stale = False

    def on_control_timer(self):
        self._jitter_tracker.log(self.get_logger().warn)

        # Wait until state is fully populated by the subscriber
        if self.robot_state.joint_actuated_values is None or self.robot_state.joint_unactuated_values is None:
            return

        if None in (
            self.robot_state.joint_actuated_values.position,
            self.robot_state.joint_actuated_values.velocity,
            self.robot_state.joint_unactuated_values.position,
            self.robot_state.joint_unactuated_values.velocity,
        ):
            self.get_logger().warn("Received None values in joint states, skipping inference.")
            return

        if not self._drive_ready:
            # Hold at the measured position so the drive's first setpoint after activation
            # is the current position (no slew), and the EMA filter is seeded from there.
            self.publish_command(self.robot_state.joint_actuated_values.position)
            return

        now = time.monotonic()
        stale = (
            self._gim_stamp is None
            or self._free_stamp is None
            or now - self._gim_stamp > STATE_STALE_TIMEOUT_S
            or now - self._free_stamp > STATE_STALE_TIMEOUT_S
        )
        if stale:
            if not self._stale:
                self.get_logger().error(
                    f"Joint state stale (gim age: "
                    f"{'n/a' if self._gim_stamp is None else f'{now - self._gim_stamp:.3f}'}s, free age: "
                    f"{'n/a' if self._free_stamp is None else f'{now - self._free_stamp:.3f}'}s) "
                    "- holding motor at current position.",
                    throttle_duration_sec=1.0,
                )
            self._stale = True
            self.publish_command(self.robot_state.joint_actuated_values.position)
            return

        if self._stale:
            self.get_logger().info("Joint state fresh again - resetting inference pipeline.")
            self._stale = False
            self._reset_pipeline()
        elif not self._seeded:
            self._reset_pipeline()
        self._seeded = True

        output = self._inference_pipeline.infer(PendulumPipelineInput(
            gim_pos=self.robot_state.joint_actuated_values.position,
            gim_vel=self.robot_state.joint_actuated_values.velocity,
            free_pos=self.robot_state.joint_unactuated_values.position,
            free_vel=self.robot_state.joint_unactuated_values.velocity,
        ))

        self.publish_command(output.gim_pos)

    def _reset_pipeline(self):
        """Clear obs history and reseed the EMA filter from the measured motor position."""
        self._inference_pipeline.reset()
        self._inference_pipeline.output.reset(self.robot_state.joint_actuated_values.position)

    def _check_controller_active(self):
        if self._drive_ready or not self._list_controllers_cli.service_is_ready():
            return
        future = self._list_controllers_cli.call_async(ListControllers.Request())
        future.add_done_callback(self._on_controller_list)

    def _on_controller_list(self, future):
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"list_controllers call failed: {exc}")
            return
        for controller in response.controller:
            if controller.name == POSITION_CONTROLLER_NAME and controller.state == "active":
                self._drive_ready = True
                self._controller_check_timer.cancel()
                self.get_logger().info(
                    f"{POSITION_CONTROLLER_NAME} is active - drive in closed loop, starting inference."
                )
                return

    def publish_command(self, target_position: float):
        msg = Float64MultiArray()
        msg.data = [target_position]
        self.cmd_pub.publish(msg)

    def on_joint_state(self, msg: DynamicJointState):
        now = time.monotonic()
        for joint_name in msg.joint_names:
            if joint_name == JOINT_ACTUATED:
                self._gim_stamp = now
            elif joint_name == JOINT_UNACTUATED:
                self._free_stamp = now
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