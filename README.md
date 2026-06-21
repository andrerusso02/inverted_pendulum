# inverted Pendulum

## Setup

```bash
cd inverted_pendulum
sudo ln -s $(pwd)/system/canable.udev /etc/udev/rules.d/99-canable.rules
sudo ln -s $(pwd)/system/canable.service /etc/systemd/system/canable.service

sudo systemctl daemon-reload
sudo udevadm control --reload-rules
sudo udevadm trigger
```

# Before running

```bash
sudo nmcli device set enx00005e005301 managed no && sudo ip addr add dev enx00005e005301 192.0.2.2/24
```
