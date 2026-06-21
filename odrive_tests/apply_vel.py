import odrive
import odrive.enums

odrv0 = odrive.find_any()

odrv0.axis0.requested_state = odrive.enums.AxisState.CLOSED_LOOP_CONTROL
odrv0.axis0.controller.config.control_mode = odrive.enums.ControlMode.VELOCITY_CONTROL

odrv0.axis0.controller.input_vel = 1.5  # rotation per second of input shaft

while True:
    print(odrv0.axis0.encoder.vel_estimate)  # rotation per second of input shaft