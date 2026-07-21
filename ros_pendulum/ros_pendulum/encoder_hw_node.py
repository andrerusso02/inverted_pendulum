import rclpy
from rclpy.node import Node
from control_msgs.msg import DynamicJointState, InterfaceValue
from sensor_msgs.msg import JointState
import socket
import struct
import math
from ros_pendulum.ping_checker import PingChecker

class AS5600JointStateNode(Node):
    def __init__(self):
        super().__init__('as5600_joint_state_node')
        
        # 1. Declare and retrieve ROS parameters for dynamic configuration
        self.declare_parameter('device_ip', '192.0.2.1')
        self.declare_parameter('bind_ip', '0.0.0.0')
        self.declare_parameter('udp_port', 5005)
        self.declare_parameter('joint_name', 'free_0')
        
        self.device_ip = self.get_parameter('device_ip').value
        self.bind_ip = self.get_parameter('bind_ip').value
        self.udp_port = self.get_parameter('udp_port').value
        self.joint_name = self.get_parameter('joint_name').value
        
        self.dynamic_joint_pub = self.create_publisher(DynamicJointState, '/dynamic_joint_states', 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        self.ping_checker = PingChecker(
            ip_address=self.device_ip,
            log_func=self.get_logger().warn,
            interval=1.0
        )
        self.ping_checker.start()
        
        self.ping_checker.block_until_ping_ok()
        self.get_logger().info(f"Ping to {self.device_ip} successful. Starting node.")
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.bind_ip, self.udp_port))
        self.sock.setblocking(False) 
        
        self.timer = self.create_timer(0.01, self.update)
        self.last_received_time = self.get_clock().now()
        
        self.get_logger().info(
            f"Initialized. Listening on {self.bind_ip}:{self.udp_port} for joint '{self.joint_name}'."
        )

    def update(self):
        self.poll_udp()
        
        
    def build_dynamic_joint_state_msg(self, radians: float) -> DynamicJointState:
        msg = DynamicJointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [self.joint_name]
        
        interface_val = InterfaceValue()
        interface_val.interface_names = ['position']
        interface_val.values = [float(radians)]
        
        msg.interface_values = [interface_val]
        
        return msg

    def build_joint_state_msg(self, radians: float) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self.joint_name]
        msg.position = [float(radians)]
        return msg

    def poll_udp(self):
        try:
            data, _ = self.sock.recvfrom(8)
            
            if len(data) == 8:
                val1, val2 = struct.unpack('<ii', data)
                
                degrees = val1 + (val2 / 1_000_000.0)
                radians = math.radians(degrees)
                
                dynamic_joint_state_msg = self.build_dynamic_joint_state_msg(radians)
                joint_state_msg = self.build_joint_state_msg(radians)

                self.dynamic_joint_pub.publish(dynamic_joint_state_msg)
                self.joint_pub.publish(joint_state_msg)
                self.last_received_time = self.get_clock().now()
            
            else:
                self.get_logger().warn(f"Received unexpected data length: {len(data)} bytes")
                
        except BlockingIOError:
            t_no_data_s = (self.get_clock().now() - self.last_received_time).nanoseconds / 1e9
            if t_no_data_s > 2.0:
                self.get_logger().warn(f"No data received for over {t_no_data_s:.2f} seconds.", throttle_duration_sec=2.0)

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
        rclpy.shutdown()

if __name__ == '__main__':
    main()