import odrive
import odrive.enums

import time

odrv = odrive.find_any()

# odrv.axis0.requested_state = odrive.enums.AxisState.CLOSED_LOOP_CONTROL
# odrv.axis0.controller.config.control_mode = odrive.enums.ControlMode.POSITION_CONTROL

while True:
    # print(F"Position: {odrv.axis0.encoder.pos_estimate}")  # input shaft turns
    print(F"Torque: {odrv.axis0.motor.current_control.Iq_measured}")  # we should multiply by torque constant to get torque in Nm
    time.sleep(0.1)
