# inverted Pendulum

## Tooling
```bash
sudo apt install can-utils
pip3 install odrive --break-system-packages
```

## Setup

```bash
cd ~/git
git clone git@github.com:andrerusso02/inverted_pendulum.git
git clone git@github.com:andrerusso02/ros_odrive.git

mkdir -p ~/pendulum_ws/src
ln -s ~/git/inverted_pendulum/ros_pendulum pendulum_ws/src/
ln -s ~/git/ros_odrive pendulum_ws/src/
```

```bash
cd inverted_pendulum
sudo ln -s $(pwd)/system/odrive.udev /etc/udev/rules.d/99-odrive.rules
sudo ln -s $(pwd)/system/canable.udev /etc/udev/rules.d/99-canable.rules
sudo ln -s $(pwd)/system/canable.service /etc/systemd/system/canable.service


sudo systemctl daemon-reload
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Build ROS packages
```bash
cd ~/pendulum_ws
rosdep install --from-paths src -y --ignore-src
colcon build --packages-up-to ros_pendulum
```

## Before running

```bash
sudo nmcli device set enx00005e005301 managed no && sudo ip addr add dev enx00005e005301 192.0.2.2/24
```
or set it up permanently:
```bash
sudo nmcli connection add type ethernet ifname enx00005e005301 con-name encoder_eth ipv4.method manual ipv4.addresses 192.0.2.2/24
```

## Control the motor manually:
```bash
ros2 topic pub /joint_group_position_controller/commands std_msgs/msg/Float64MultiArray 'layout:
  dim:
    - label: "axis"
      size: 1
      stride: 1
  data_offset: 0
data: [4.0]
```

# IsaacLab

Add this to activate script of isaaclab env:
```bash
source ~/git/inverted_pendulum/Pendulum/env.bash
```

# Odrive

Backup settings:
```bash
odrivetool backup-config config.json
```

Flash settings:
```bash
odrivetool restore-config config.json
```
