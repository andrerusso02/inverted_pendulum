import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import socket
import struct
import math
import os
from ros_pendulum.ping_checker import PingChecker

DEVICE_IP_ADDRESS = '192.0.2.1'
BIND_IP_ADDRESS = '0.0.0.0'
UDP_PORT = 5005

class AS5600JointStateNode(Node):
    def __init__(self):
        super().__init__('as5600_joint_state_node')
        
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        self.ping_checker = PingChecker(
            ip_address=DEVICE_IP_ADDRESS,
            log_func=self.get_logger().warn,
            interval=1.0
        )
        self.ping_checker.start()
        
        self.ping_checker.block_until_ping_ok()
        self.get_logger().info(f"Ping to {DEVICE_IP_ADDRESS} successful. Starting node.")
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((BIND_IP_ADDRESS, UDP_PORT))
        self.sock.setblocking(False) 
        
        self.timer = self.create_timer(0.01, self.update)
        

        
        self.joint_name = 'free_0'
        
        self.last_received_time = self.get_clock().now()
        
        self.get_logger().info(
            f"AS5600 Joint State Node initialized and listening on {BIND_IP_ADDRESS}:{UDP_PORT}."
        )
        
    
    def update(self):
        self.poll_udp()

    def poll_udp(self):
        try:
            # 2x int32 = 8 bytes
            data, _ = self.sock.recvfrom(8)
            
            if len(data) == 8:
                # Unpack two 32-bit integers ('<ii' specifies little-endian)
                val1, val2 = struct.unpack('<ii', data)
                
                # Combine degrees and micro-degrees, then convert to radians
                degrees = val1 + (val2 / 1_000_000.0)
                radians = math.radians(degrees)
                
                # Populate JointState Message
                joint_msg = JointState()
                joint_msg.header.stamp = self.get_clock().now().to_msg()
                joint_msg.name = [self.joint_name]
                joint_msg.position = [radians]
                
                # Publish the message
                self.joint_pub.publish(joint_msg)
                
                self.last_received_time = self.get_clock().now()
            
            else:
                self.get_logger().warn(f"Received unexpected data length: {len(data)} bytes")
                
        except BlockingIOError:
            t_no_data_s = (self.get_clock().now() - self.last_received_time).nanoseconds / 1_000_000_000.0
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