#!/usr/bin/env python3

"""Record UR5e TCP wrench through RTDE receive interface."""

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
            "force at a fixed rate, and plot the results when stopped."
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
    forces: List[List[float]] = []

    print(
        "Sampling TCP force (Fx, Fy, Fz, Mx, My, Mz). Press Ctrl+C to stop and plot."
    )

    start_time = time.monotonic()
    try:
        while True:
            loop_start = time.monotonic()
            try:
                tcp_force = rtde_r.getActualTCPForce()
            except Exception as exc:  # pragma: no cover - hardware dependent
                print(f"Error fetching TCP force: {exc}")
                break

            if tcp_force is None:
                time.sleep(sample_period)
                continue

            times.append(loop_start - start_time)
            forces.append(list(tcp_force))

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

    if not forces:
        print("No force data captured.")
        return 0

    forces_arr = np.asarray(forces)
    times_arr = np.asarray(times)

    max_vals = forces_arr.max(axis=0)
    max_abs_vals = np.abs(forces_arr).max(axis=0)

    component_labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    print("Maximum TCP force components:")
    for label, value, abs_value in zip(component_labels, max_vals, max_abs_vals):
        print(f"  {label}: max={value:.3f}, max|.|={abs_value:.3f}")

    fig, (ax_force, ax_torque) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    force_components = forces_arr[:, :3]
    torque_components = forces_arr[:, 3:]

    for idx, label in enumerate(component_labels[:3]):
        ax_force.plot(times_arr, force_components[:, idx], label=label)
    ax_force.set_ylabel("Force (N)")
    ax_force.legend(loc="upper right")
    ax_force.grid(True, linestyle="--", alpha=0.4)

    for idx, label in enumerate(component_labels[3:]):
        ax_torque.plot(times_arr, torque_components[:, idx], label=label)
    ax_torque.set_ylabel("Torque (Nm)")
    ax_torque.set_xlabel("Time (s)")
    ax_torque.legend(loc="upper right")
    ax_torque.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("UR5e TCP Force and Torque vs Time")
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())



