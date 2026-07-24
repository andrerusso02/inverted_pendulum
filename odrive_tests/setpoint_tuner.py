import sys
import tty
import termios
import odrive
import odrive.enums
from odrive.utils import dump_errors

import math

def get_key():
    """Reads a single keypress from standard input."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # Arrow keys send an escape sequence starting with \x1b
        if ch == '\x1b':
            ch2 = sys.stdin.read(2)
            if ch2 == '[A':
                return 'up'
            elif ch2 == '[B':
                return 'down'
            return 'esc'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

print("Finding ODrive...")
odrv0 = odrive.find_any()

print("Clearing errors...")
dump_errors(odrv0, True)
dump_errors(odrv0)

print("Configuring axis...")
odrv0.axis0.requested_state = odrive.enums.AxisState.CLOSED_LOOP_CONTROL
odrv0.axis0.controller.config.control_mode = odrive.enums.ControlMode.POSITION_CONTROL
odrv0.axis0.controller.config.input_mode = odrive.enums.InputMode.PASSTHROUGH

odrv0.axis0.motor.config.torque_lim = 0.50

SETPOINT_RAD = 0.0
SETPOINT_INCREMENT = 0.1
GEAR_RATIO = 8.0

print("Ready. Use UP and DOWN arrows to adjust position. Press 'q' or ESC to quit.")

# Set initial position
odrv0.axis0.controller.input_pos = SETPOINT_RAD * GEAR_RATIO

while True:
    key = get_key()
    
    if key == 'up':
        SETPOINT_RAD += SETPOINT_INCREMENT
    elif key == 'down':
        SETPOINT_RAD -= SETPOINT_INCREMENT
    elif key in ('esc', 'q', '\x03'):  # \x03 catches Ctrl+C
        print("Exiting...")
        break
    else:
        continue
        
    input_shaft_setpoint_rad = (SETPOINT_RAD * GEAR_RATIO) 
    odrv0.axis0.controller.input_pos = input_shaft_setpoint_rad / (2*math.pi)
    print(f"Setpoint: {SETPOINT_RAD:.1f} | Moving to motor position: {input_shaft_setpoint_rad:.2f}")

# Set motor back to idle upon exit
odrv0.axis0.requested_state = odrive.enums.AxisState.IDLE