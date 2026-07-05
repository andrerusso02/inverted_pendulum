import odrive
import odrive.enums
import time
from odrive.utils import dump_errors

odrv0 = odrive.find_any()

dump_errors(odrv0, True)
dump_errors(odrv0)

odrv0.axis0.requested_state = odrive.enums.AxisState.CLOSED_LOOP_CONTROL
odrv0.axis0.controller.config.control_mode = odrive.enums.ControlMode.POSITION_CONTROL
odrv0.axis0.controller.config.input_mode = odrive.enums.InputMode.POS_FILTER

# odrv0.axis0.trap_traj.config.vel_limit = 10.

# odrv0.axis0.controller.config.pos_gain = 3.0
# odrv0.axis0.controller.config.vel_gain = 0.005
# odrv0.axis0.controller.config.vel_integrator_gain = 0.05

# odrv0.axis0.controller.config.input_filter_bandwidth = 20.0

# odrv0.axis0.motor.config.torque_lim = 2.0

POS0 = 0.0
POS1 = 2.0

while True:
    odrv0.axis0.controller.input_pos = POS0
    print(f"Moving to position {POS0}")
    time.sleep(2)
    odrv0.axis0.controller.input_pos = POS1
    print(f"Moving to position {POS1}")
    time.sleep(2)
