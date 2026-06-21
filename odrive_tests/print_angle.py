import odrive
import odrive.enums

odrv = odrive.find_any()

# disable torque on axis 0
odrv.axis0.requested_state = odrive.enums.AXIS_STATE_IDLE

while True:
    print(odrv.axis0.encoder.pos_estimate)  # input shaft turns