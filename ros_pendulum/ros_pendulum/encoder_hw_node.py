import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import socket
import struct
import math

class AS5600JointStateNode(Node):
    def __init__(self):
        super().__init__('as5600_joint_state_node')
        
        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Set up UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 5005))
        self.sock.setblocking(False) 
        
        # Poll socket at 100Hz
        self.timer = self.create_timer(0.01, self.poll_udp)
        
        # Configuration
        self.joint_name = 'free_0'

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
                
        except BlockingIOError:
            pass 

def main(args=None):
    rclpy.init(args=args)
    node = AS5600JointStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()