
import odrive

odrv = odrive.find_any()

print(odrv.axis0.config.can.node_id)