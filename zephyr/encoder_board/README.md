# For succesful build

Update line...
```
project(second_stage_bootloader C ASM)

```
... in `zephyr/modules/hal_rpi_pico/bootloader/CMakeLists.txt`

# Build
```
west build -p always -b rpi_pico ~/git/inverted_pendulum/zephyr/encoder_board
```

# Flash

```
sudo cp build/zephyr/zephyr.uf2 /media/andre/RPI-RP2/
```