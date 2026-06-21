import json
import odrive
from odrive.utils import backup_config

odrv = odrive.find_any()

config_dict = backup_config(odrv)

print(json.dumps(config_dict, indent=4))
