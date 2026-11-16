#!/usr/bin/env python3
"""Wait for the first message on /joint_group_position_controller/commands.

Exits 0 and prints the payload when a message arrives; exits 1 after --timeout seconds.
Unlike `ros2 topic echo --once`, this does not fail when the topic is not yet advertised:
it subscribes unconditionally and lets discovery catch up.
"""

import argparse
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

TOPIC = "/joint_group_position_controller/commands"


class WaitForFirstMessage(Node):
    def __init__(self):
        super().__init__("wait_for_first_message")
        self.payload = None
        self.create_subscription(Float64MultiArray, TOPIC, self.on_msg, 10)

    def on_msg(self, msg):
        if self.payload is None:
            self.payload = list(msg.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    rclpy.init()
    node = WaitForFirstMessage()
    try:
        elapsed = 0.0
        step = 0.1
        while node.payload is None and elapsed < args.timeout:
            rclpy.spin_once(node, timeout_sec=step)
            elapsed += step
    except KeyboardInterrupt:
        pass
    finally:
        payload = node.payload
        node.destroy_node()
        rclpy.shutdown()

    if payload is None:
        print(f"no message on {TOPIC} within {args.timeout}s", file=sys.stderr)
        sys.exit(1)
    print(f"first command on {TOPIC}: {payload}")
    sys.exit(0)


if __name__ == "__main__":
    main()
