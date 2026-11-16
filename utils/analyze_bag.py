#!/usr/bin/env python3
"""Analyze a pendulum deployment bag: swing-up/capture, gentleness, stream health.

Usage (with ROS jazzy sourced):
    python3 utils/analyze_bag.py <bag_dir>

Reports per-0.2s aligned series (cmd, gim pos/vel/torque, free pos/vel, tilt-from-upright),
peak stats, and sustained-capture detection (<15 deg held with <25 deg for ~1 s).
"""

import bisect
import math
import sys

from control_msgs.msg import DynamicJointState
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from std_msgs.msg import Float64MultiArray


def load(bag_dir):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="mcap"),
        ConverterOptions("cdr", "cdr"),
    )
    cmds, gims, frees = [], [], []
    t0 = None
    while reader.has_next():
        topic, data, t = reader.read_next()
        if t0 is None:
            t0 = t
        rel = (t - t0) / 1e9
        if topic.endswith("/commands"):
            m = deserialize_message(data, Float64MultiArray)
            cmds.append((rel, m.data[0]))
        elif topic == "/dynamic_joint_states":
            m = deserialize_message(data, DynamicJointState)
            for jn, iv in zip(m.joint_names, m.interface_values):
                d = dict(zip(iv.interface_names, iv.values))
                if jn == "gim6010":
                    gims.append((rel, d.get("position"), d.get("velocity"), d.get("effort")))
                elif jn == "free_0":
                    frees.append((rel, d.get("position"), d.get("velocity")))
    return cmds, gims, frees


def nearest(series, ts_sorted, t):
    i = bisect.bisect_left(ts_sorted, t)
    i = min(max(i - 1, 0), len(series) - 1)
    j = min(i + 1, len(series) - 1)
    return series[i] if abs(series[i][0] - t) < abs(series[j][0] - t) else series[j]


def tilt_deg(free_pos):
    return math.degrees(abs(abs(free_pos) - math.pi))


def main(bag_dir):
    cmds, gims, frees = load(bag_dir)
    for name, s in (("cmds", cmds), ("gim", gims), ("free", frees)):
        gaps = sum(1 for a, b in zip(s, s[1:]) if b[0] - a[0] > 0.1)
        print(f"{name}: n={len(s)} t={s[0][0]:.2f}..{s[-1][0]:.2f} gaps>0.1s: {gaps}")

    cmd_t = [x[0] for x in cmds]
    gim_t = [x[0] for x in gims]
    free_t = [x[0] for x in frees]

    print(f"\n{'t':>5} {'cmd':>7} {'gim_p':>7} {'gim_v':>7} {'tau':>5} {'free_p':>7} {'free_v':>7} {'tilt°':>6}")
    tend = min(cmds[-1][0], gims[-1][0], frees[-1][0])
    t = 0.0
    while t <= tend:
        c = nearest(cmds, cmd_t, t)
        g = nearest(gims, gim_t, t)
        f = nearest(frees, free_t, t)
        tau = g[3] if g[3] is not None else float("nan")
        fv = f[2] if f[2] is not None else float("nan")
        print(f"{t:5.2f} {c[1]:7.3f} {g[1]:7.3f} {g[2]:7.2f} {tau:5.2f} {f[1]:7.3f} {fv:7.2f} {tilt_deg(f[1]):6.1f}")
        t += 0.2

    gv = [abs(x[2]) for x in gims if x[2] is not None]
    gt = [abs(x[3]) for x in gims if x[3] is not None]
    fv = [abs(x[2]) for x in frees if x[2] is not None]
    tilts = [tilt_deg(x[1]) for x in frees]
    print(f"\npeaks: gim|vel| {max(gv):.1f} rad/s | tau {max(gt):.2f} Nm | pole|vel| {max(fv):.1f} rad/s")
    print(f"tilt: best {min(tilts):.1f} deg, final {tilts[-1]:.1f} deg")

    # sustained capture: tilt <15 deg and stays <25 deg for the next ~1 s
    horizon = max(1, int(1.0 / ((frees[-1][0] - frees[0][0]) / len(frees))))
    cap = None
    for i in range(len(frees) - horizon):
        if tilt_deg(frees[i][1]) < 15 and all(tilt_deg(x[1]) < 25 for x in frees[i:i + horizon]):
            cap = frees[i][0]
            break
    if cap is None:
        print("sustained capture: NONE")
        return
    bal_f = [x for x in frees if x[0] >= cap]
    bal_g = [x for x in gims if x[0] >= cap and x[3] is not None]
    print(f"sustained capture at t={cap:.2f}s ({tend - cap:.2f}s of balance):")
    print(
        f"  tilt mean {sum(tilt_deg(x[1]) for x in bal_f) / len(bal_f):.1f} deg | "
        f"gim|vel| mean {sum(abs(x[2]) for x in bal_g) / len(bal_g):.2f} rad/s | "
        f"tau mean {sum(abs(x[3]) for x in bal_g) / len(bal_g):.2f} Nm"
    )


if __name__ == "__main__":
    main(sys.argv[1])
