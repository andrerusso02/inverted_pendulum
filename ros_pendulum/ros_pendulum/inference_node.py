import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class StaticMotorController(Node):
    def __init__(self):
        super().__init__('static_motor_controller')
        
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, 
            '/joint_group_position_controller/commands', 
            10
        )
        
        self.state_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.state_callback, 
            10
        )
        
        self.timer = self.create_timer(0.1, self.update)
        
        self.target_position = 1.0 
        
    
    def update(self):
        current_time = self.get_clock().now().to_msg()
        self.target_position = 1.0 if current_time.sec % 2 == 0 else 0.0
        
        self.publish_command()

    def publish_command(self):
        msg = Float64MultiArray()
        msg.data = [self.target_position]
        self.cmd_pub.publish(msg)

    def state_callback(self, msg: JointState):
        try:
            joint_idx = msg.name.index('gim6010')
            current_pos = msg.position[joint_idx]
            
            self.get_logger().info(
                f"Target: {self.target_position:.4f} | Current: {current_pos:.4f}"
            )
        except ValueError:
            self.get_logger().warn("'gim6010' not found in joint_states", once=True)
        except IndexError:
            self.get_logger().error("Joint found, but position array is empty")

def main(args=None):
    rclpy.init(args=args)
    node = StaticMotorController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()