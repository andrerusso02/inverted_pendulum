from enum import StrEnum
from dataclasses import dataclass
import rclpy
from typing import Any
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import DynamicJointState

JOINT_ACTUATED = 'gim6010'
JOINT_UNACTUATED = 'free_0'

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


class StaticMotorController(Node):
    def __init__(self):
        super().__init__('static_motor_controller')
        
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
        
        self.timer = self.create_timer(0.1, self.update)
        
        self.target_position = 1.0 
        
        self.robot_state = RobotState()
        
    
    def update(self):
        current_time = self.get_clock().now().to_msg()
        self.target_position = 1.0 if current_time.sec % 2 == 0 else 0.0
        
        self.publish_command()

    def publish_command(self):
        msg = Float64MultiArray()
        msg.data = [self.target_position]
        self.cmd_pub.publish(msg)


    def on_joint_state(self, msg: DynamicJointState):
        self.robot_state.update_from_joint_state(msg)
        self.get_logger().info(f"{self.robot_state}")

        
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