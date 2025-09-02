from typing import Any, Optional

import numpy as np
from loguru import logger

from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import RobotConfigStatus
from phosphobot.utils import get_resources_path


class UR5eHardware(BaseManipulator):
    name = "ur5e"

    URDF_FILE_PATH = str(
        get_resources_path() / "urdf" / "ur5e" / "urdf" / "ur5e.urdf"
    )

    # World orientation (x, y, z, w) quaternion used to place the base in sim
    AXIS_ORIENTATION = [0, 0, 0, 1]

    # Default: no gripper in URDF
    END_EFFECTOR_LINK_INDEX = 8  # will be corrected at runtime if needed
    GRIPPER_JOINT_INDEX = -1

    # Kinematic resolution placeholders (unused for RTDE path)
    RESOLUTION = 4096
    SERVO_IDS = [0, 1, 2, 3, 4, 5]
    CALIBRATION_POSITION = [0.0, -1.57, 1.57, -1.57, 1.57, 0.0]

    def __init__(
        self,
        only_simulation: bool = False,
        host: str = "192.168.0.2",
        speed: float = 0.3,
        acc: float = 0.6,
        **kwargs: Any,
    ) -> None:
        # Store HW params first
        self.only_simulation = only_simulation
        self.host = host
        self.speed = float(speed)
        self.acc = float(acc)

        # RTDE objects
        self.ctrl = None
        self.recv = None

        # Initialize simulation (loads URDF, sets self.p_robot_id, self.actuated_joints)
        super().__init__(only_simulation=True if only_simulation else False, **kwargs)

        # Try to auto-detect the end-effector link index by name when possible
        try:
            import pybullet as p  # type: ignore

            num_joints = p.getNumJoints(self.p_robot_id)
            ee_candidates = {b"tool0", b"flange", b"wrist_3_link"}
            for i in range(num_joints):
                info = self.sim.get_joint_info(self.p_robot_id, i)
                link_name = info[12] if len(info) > 12 else b""
                if link_name in ee_candidates:
                    self.END_EFFECTOR_LINK_INDEX = i
            logger.debug(
                f"UR5e END_EFFECTOR_LINK_INDEX set to {self.END_EFFECTOR_LINK_INDEX}"
            )
        except Exception:
            pass

    async def connect(self) -> None:
        if self.only_simulation:
            self.is_connected = False
            logger.info("UR5e running in simulation-only mode; skipping RTDE connect.")
            return

        try:
            # Lazy import to avoid hard dependency when sim-only
            from ur_rtde import rtde_control, rtde_receive  # type: ignore

            self.ctrl = rtde_control.RTDEControlInterface(self.host)
            self.recv = rtde_receive.RTDEReceiveInterface(self.host)
            self.is_connected = True
            self.init_config()
            logger.success(f"Connected to UR5e at {self.host}")
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to UR5e at {self.host}: {e}")
            raise

    def disconnect(self) -> None:
        try:
            if self.ctrl is not None:
                self.ctrl.disconnect()
        except Exception:
            pass
        try:
            if self.recv is not None:
                self.recv.disconnect()
        except Exception:
            pass
        self.is_connected = False

    def status(self) -> RobotConfigStatus:
        return RobotConfigStatus(name=self.name, device_name=self.host, temperature=None)

    def read_joints_position(
        self,
        unit: str = "rad",
        source: str = "robot",
        joints_ids: Optional[list[int]] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> np.ndarray:
        # If hardware connected and source is robot, read RTDE
        if source == "robot" and self.is_connected and self.recv is not None:
            try:
                q = np.array(self.recv.getActualQ(), dtype=float)
                if unit == "degrees":
                    return np.rad2deg(q)
                return q
            except Exception as e:
                logger.warning(f"RTDE read_joints_position failed: {e}; falling back to sim")
        # Fallback to base behavior (sim)
        return super().read_joints_position(
            unit=unit, source="sim", joints_ids=joints_ids, min_value=min_value, max_value=max_value
        )

    def set_motors_positions(
        self, q_target_rad: np.ndarray, enable_gripper: bool = False
    ) -> None:
        # Send to hardware if connected
        if self.is_connected and self.ctrl is not None:
            try:
                self.ctrl.moveJ(q_target_rad.tolist(), self.speed, self.acc)
            except Exception as e:
                logger.warning(f"RTDE moveJ failed: {e}")

        # Always update simulation for visualization
        super().set_motors_positions(q_target_rad=q_target_rad, enable_gripper=False)

    async def move_to_initial_position(self) -> None:
        # Safe tuck pose
        q = np.array([0.0, -1.57, 1.57, -1.57, 1.57, 0.0], dtype=float)
        self.set_motors_positions(q_target_rad=q, enable_gripper=False)
        # Cache initial pose for absolute moves
        pos, orn = self.forward_kinematics()
        self.initial_position = pos
        self.initial_orientation_rad = orn

    async def move_to_sleep(self) -> None:
        q = np.array([0.0, -2.3, 1.8, -1.5, 1.5, 0.0], dtype=float)
        self.set_motors_positions(q_target_rad=q, enable_gripper=False)

    def enable_torque(self) -> None:
        # UR RTDE uses trajectory commands; no explicit torque enable
        return

    def disable_torque(self) -> None:
        try:
            if self.is_connected and self.ctrl is not None:
                self.ctrl.stopJ(0.5)
        except Exception:
            pass
