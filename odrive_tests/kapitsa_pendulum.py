import odrive
import odrive.enums
import time
import math
from odrive.utils import dump_errors

# Configuration Constants
CONTROL_FREQ_HZ = 100.0  # Update frequency for sending position commands
SIN_AMPLITUDE = 0.5     # Peak amplitude of the sine wave (in turns)
SIN_FREQ_HZ = 15.       # Frequency of the sine wave (cycles per second)
SIN_OFFSET = 1.0        # Center position offset (in turns)

# Connect to ODrive
odrv0 = odrive.find_any()

dump_errors(odrv0, True)
dump_errors(odrv0)

# Configure control modes
odrv0.axis0.requested_state = odrive.enums.AxisState.CLOSED_LOOP_CONTROL
odrv0.axis0.controller.config.control_mode = odrive.enums.ControlMode.POSITION_CONTROL
odrv0.axis0.controller.config.input_mode = odrive.enums.InputMode.PASSTHROUGH

odrv0.axis0.motor.config.torque_lim = 0.50

dt = 1.0 / CONTROL_FREQ_HZ
start_time = time.time()

try:
    while True:
        elapsed_time = time.time() - start_time
        
        # Calculate sine target position
        target_pos = SIN_OFFSET + SIN_AMPLITUDE * math.sin(2 * math.pi * SIN_FREQ_HZ * elapsed_time)
        
        # Send position command
        odrv0.axis0.controller.input_pos = target_pos
        
        time.sleep(dt)

except KeyboardInterrupt:
    # Safely stop motor on exit
    odrv0.axis0.requested_state = odrive.enums.AxisState.IDLE
    print("\nControl loop stopped. Axis set to IDLE.")