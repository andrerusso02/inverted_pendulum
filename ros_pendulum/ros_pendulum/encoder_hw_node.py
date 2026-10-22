import math
import socket
import struct
import threading

import rclpy
import rclpy.logging
from builtin_interfaces.msg import Time
from control_msgs.msg import DynamicJointState, InterfaceValue
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ros_pendulum.ping_checker import PingChecker

DIRECTION = -1  # 1 for clockwise, -1 for counter-clockwise


class UDPReceiver:
    def __init__(self, bind_ip: str, udp_port: int, on_data_callback):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, udp_port))
        self.sock.setblocking(True)
        self.sock.settimeout(1.0)  # Block for up to 1.0 seconds, then raise socket.timeout
        
        self.on_data_callback = on_data_callback

        self.polling_thread = threading.Thread(target=self._poll_udp_loop)
        self.polling_thread.start()
        
        self.logger = rclpy.logging.get_logger("udp_receiver_logger")
    
    def _poll_udp_loop(self):
        while True:
            data = self.poll_udp()
            if data is not None:
                receive_time = rclpy.clock.Clock().now()
                self.on_data_callback(receive_time, data)
    
    def poll_udp(self):
        try:
            data, _ = self.sock.recvfrom(8)
            return data
        except TimeoutError:
            self.logger.warning("UDP receive timed out after 1 second receiving no data.")
            return None


class VelocityEstimator:
    def __init__(self):
        self._last_pos = None
        self._last_time = None
        self._velocity = 0.0

    def update(self, angle, current_time):
        if self._last_pos is not None and self._last_time is not None:
            delta_pos = angle - self._last_pos
            delta_time = (current_time - self._last_time).nanoseconds / 1e9
            self._velocity = delta_pos / delta_time

        self._last_pos = angle
        self._last_time = current_time

    @property
    def current_velocity(self):
        return self._velocity


def wrap_around_pi(angle_rad):
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


def build_angle_from_data(val1, val2):
    degrees = val1 + (val2 / 1_000_000.0)
    radians = math.radians(degrees)
    return wrap_around_pi(radians) * DIRECTION

def build_dynamic_joint_state_msg(
    joint_name: str, stamp: Time, pos_rad: float, vel_rad_s: float
) -> DynamicJointState:
    msg = DynamicJointState()
    msg.header.stamp = stamp
    msg.joint_names = [joint_name]

    interface_val = InterfaceValue()
    interface_val.interface_names = ["position", "velocity"]
    interface_val.values = [float(pos_rad), float(vel_rad_s)]

    msg.interface_values = [interface_val]

    return msg

def build_joint_state_msg(
    joint_name: str, stamp: Time, pos_rad: float, vel_rad_s: float
) -> JointState:
    msg = JointState()
    msg.header.stamp = stamp
    msg.name = [joint_name]
    msg.position = [float(pos_rad)]
    msg.velocity = [float(vel_rad_s)]
    return msg


class AS5600JointStateNode(Node):
    def __init__(self):
        super().__init__("as5600_joint_state_node")

        self.declare_parameter("device_ip", "192.0.2.1")
        self.declare_parameter("bind_ip", "0.0.0.0")
        self.declare_parameter("udp_port", 5005)
        self.declare_parameter("joint_name", "free_0")

        self.device_ip = self.get_parameter("device_ip").value
        self.bind_ip = self.get_parameter("bind_ip").value
        self.udp_port = self.get_parameter("udp_port").value
        self.joint_name = self.get_parameter("joint_name").value

        self.dynamic_joint_pub = self.create_publisher(
            DynamicJointState, "/dynamic_joint_states", 10
        )
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)

        self.ping_checker = PingChecker(
            ip_address=self.device_ip, log_func=self.get_logger().warn, interval=1.0
        )
        self.ping_checker.start()

        self.ping_checker.block_until_ping_ok()
        self.get_logger().info(f"Ping to {self.device_ip} successful. Starting node.")

        self.velocity_estimator = VelocityEstimator()

        self.get_logger().info(
            f"Initialized. Listening on {self.bind_ip}:{self.udp_port} for joint '{self.joint_name}'."
        )
        
        self.udp_receiver = UDPReceiver(
            bind_ip=self.bind_ip,
            udp_port=self.udp_port,
            on_data_callback=self.on_udp_receive,
        )

    def on_udp_receive(self, receive_time: Time, data: bytes):

        if len(data) != 8:
            raise ValueError(f"Received data of unexpected length: {len(data)} bytes.")
                
        val1, val2 = struct.unpack("<ii", data)
        pos_rad = build_angle_from_data(val1, val2)
        
        self.velocity_estimator.update(pos_rad, receive_time)

        stamp = receive_time.to_msg()
        dynamic_joint_state_msg = build_dynamic_joint_state_msg(
            joint_name=self.joint_name,
            stamp=stamp,
            pos_rad=pos_rad,
            vel_rad_s=self.velocity_estimator.current_velocity,
        )
        joint_state_msg = build_joint_state_msg(
            joint_name=self.joint_name,
            stamp=stamp,
            pos_rad=pos_rad,
            vel_rad_s=self.velocity_estimator.current_velocity,
        )

        self.dynamic_joint_pub.publish(dynamic_joint_state_msg)
        self.joint_pub.publish(joint_state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AS5600JointStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ping_checker.stop()
        node.destroy_node()


if __name__ == "__main__":
    main()
