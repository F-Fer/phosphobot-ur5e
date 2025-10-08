#!/usr/bin/env python3

"""Record UR5e TCP speed through RTDE receive interface."""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import rtde_receive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to a UR5e via the RTDE receive interface, sample the TCP "
            "speed at a fixed rate, and plot the results when stopped."
        )
    )
    parser.add_argument(
        "--host",
        default="192.168.1.10",
        help="UR controller hostname or IP (default: 192.168.1.10)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=10.0,
        help="Sampling frequency in Hz (default: 10)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.frequency <= 0:
        print("Frequency must be positive.")
        return 1

    sample_period = 1.0 / args.frequency

    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(args.host)
    except Exception as exc:  # pragma: no cover - hardware dependent
        print(f"Failed to connect to RTDE receive interface: {exc}")
        return 2

    times: List[float] = []
    speeds: List[List[float]] = []

    print(
        "Sampling TCP speed (vx, vy, vz, wx, wy, wz). Press Ctrl+C to stop and plot."
    )

    start_time = time.monotonic()
    try:
        while True:
            loop_start = time.monotonic()
            try:
                tcp_speed = rtde_r.getActualTCPSpeed()
            except Exception as exc:  # pragma: no cover - hardware dependent
                print(f"Error fetching TCP speed: {exc}")
                break

            if tcp_speed is None:
                time.sleep(sample_period)
                continue

            times.append(loop_start - start_time)
            speeds.append(list(tcp_speed))

            elapsed = time.monotonic() - loop_start
            sleep_time = sample_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        try:
            rtde_r.disconnect()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    if not speeds:
        print("No speed data captured.")
        return 0

    speeds_arr = np.asarray(speeds)
    times_arr = np.asarray(times)

    linear_mm = speeds_arr[:, :3] * 1000.0
    linear_norm_mm = np.linalg.norm(linear_mm, axis=1)

    component_labels = ["vx", "vy", "vz"]

    max_components_mm = linear_mm.max(axis=0)
    max_abs_components_mm = np.abs(linear_mm).max(axis=0)
    max_norm_mm = linear_norm_mm.max()

    print("Maximum TCP linear speed components (mm/s):")
    for label, value, abs_value in zip(component_labels, max_components_mm, max_abs_components_mm):
        print(f"  {label}: max={value:.2f}, max|.|={abs_value:.2f}")
    print(f"Maximum TCP linear speed magnitude (mm/s): {max_norm_mm:.2f}")

    fig, (ax_lin, ax_norm) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for idx, label in enumerate(component_labels):
        ax_lin.plot(times_arr, linear_mm[:, idx], label=label)
    ax_lin.set_ylabel("Linear speed (mm/s)")
    ax_lin.legend(loc="upper right")
    ax_lin.grid(True, linestyle="--", alpha=0.4)

    ax_norm.plot(times_arr, linear_norm_mm, label="|v|")
    ax_norm.set_ylabel("Speed magnitude (mm/s)")
    ax_norm.set_xlabel("Time (s)")
    ax_norm.legend(loc="upper right")
    ax_norm.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("UR5e TCP Linear Speed vs Time")
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())


