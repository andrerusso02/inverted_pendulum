import odrive
import odrive.enums

odrv0 = odrive.find_any()

odrv0.axis0.requested_state = odrive.enums.AxisState.CLOSED_LOOP_CONTROL
odrv0.axis0.controller.config.control_mode = odrive.enums.ControlMode.TORQUE_CONTROL

odrv0.axis0.controller.input_torque = -0.2  # N.m of the input shaft
