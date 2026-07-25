See [Documentation](https://onshape-to-robot.readthedocs.io/en/latest/getting_started.html)

```bash
pip install onshape-to-robot
cd Pendulum/robot
onshape-to-robot config.json
```

In robot.urdf, replace every `<limit>` tag by:
```xml
    <limit lower="-100000.0" upper="100000.0" effort="100000.0" velocity="100000.0"/>
```
This is needed to disable PhysX -2pi -> 2pi wraparound and make the joint angle continuous


