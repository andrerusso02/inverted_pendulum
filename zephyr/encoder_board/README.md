# Patch for successful build

This was not needed on another computer with up to date Zephyr and SDK.

Update line...
```bash
project(second_stage_bootloader C ASM)
```
... in `zephyr/modules/hal_rpi_pico/bootloader/CMakeLists.txt`

# Build
```bash
west build -p always -b rpi_pico ~/git/inverted_pendulum/zephyr/encoder_board
```

# Flash

```bash
west flash --runner uf2
```

If encountering issues with this method, transfer the file manually:

```bash
sudo cp build/zephyr/zephyr.uf2 /media/andre/RPI-RP2/
```
