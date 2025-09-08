from typing import Any, Optional, List, Literal, Tuple

import asyncio
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R
import rtde_control
import rtde_receive
import rtde_io

from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import RobotConfigStatus
from phosphobot.models.lerobot_dataset import BaseRobotInfo, FeatureDetails
from phosphobot.utils import get_resources_path
from phosphobot.hardware.utils.robotiq_gripper import RobotiqGripper


class UR5eHardware(BaseManipulator):
    name = "ur5e"

    # URDF and kinematic indices for simulation
    URDF_FILE_PATH = str(get_resources_path() / "urdf" / "ur5e" / "urdf" / "ur5e.urdf")
    AXIS_ORIENTATION = [0, 0, 0, 1]
    END_EFFECTOR_LINK_INDEX = 6  # tool0 link in our URDF chain
    GRIPPER_JOINT_INDEX = -1     # no simulated gripper joint

    SERVO_IDS = [1, 2, 3, 4, 5, 6]
    RESOLUTION = 4096

    def __init__(
        self,
        ip: str = "192.168.1.10",
        speed: float = 0.5,
        acc: float = 0.5,
        only_simulation: bool = False,
        **kwargs: Any,
    ) -> None:
        # Base init sets up pybullet using URDF
        super().__init__(only_simulation=only_simulation, **kwargs)

        # RTDE fields (hardware side)
        self.robot_ip = ip
        self.rtde_ctrl: Optional[rtde_control.RTDEControlInterface] = None
        self.rtde_rec: Optional[rtde_receive.RTDEReceiveInterface] = None
        self.rtde_inout: Optional[rtde_io.RTDEIOInterface] = None

        # Joint-space moveJ parameters
        self.speed = float(speed)  # rad/s
        self.acc = float(acc)      # rad/s^2

        # Gripper (Robotiq on UR controller tool comms)
        self.with_gripper = True
        self.gripper = RobotiqGripper()
        self.gripper_speed = 255
        self.gripper_force = 255

        # Derived
        self.num_actuated_joints = len(self.SERVO_IDS)

    async def connect(self) -> None:
        if self.is_connected:
            return
        try:
            self.rtde_ctrl = rtde_control.RTDEControlInterface(self.robot_ip)
            self.rtde_rec = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self.rtde_inout = rtde_io.RTDEIOInterface(self.robot_ip)
            # Gripper uses UR dashboard/remote socket on 63352 by default
            try:
                self.gripper.connect(self.robot_ip, 63352)
                self.gripper.activate()
            except Exception as e:
                logger.warning(f"Robotiq gripper init failed: {e}")
            self.is_connected = True
            logger.info("UR5e RTDE connected.")
        except Exception as e:
            logger.error(f"UR5e RTDE connect error: {e}")
            self.is_connected = False
            raise

    def disconnect(self) -> None:
        try:
            if self.rtde_ctrl:
                self.rtde_ctrl.disconnect()
            if self.rtde_rec:
                self.rtde_rec.disconnect()
            if self.rtde_inout:
                self.rtde_inout.disconnect()
        finally:
            self.is_connected = False
            logger.info("UR5e RTDE disconnected.")

    # BaseManipulator abstract method impls not used for UR (feetech-style), keep minimal stubs
    def enable_torque(self) -> None:
        pass

    def disable_torque(self) -> None:
        pass

    def read_motor_torque(self, servo_id: int) -> Optional[float]:
        """
        Read the torque of a motor

        raise: Exception if the routine has not been implemented
        """
        pass

    def read_motor_voltage(self, servo_id: int) -> Optional[float]:
        """
        Read the voltage of a motor

        raise: Exception if the routine has not been implemented
        """
        pass

    def write_motor_position(self, servo_id: int, units: int, **kwargs: Any) -> None:
        # Not used; we control joints via RTDE moveJ with full vectors
        pass

    def read_motor_position(self, servo_id: int, **kwargs: Any) -> Optional[int]:
        # Not used
        return None

    def calibrate_motors(self, **kwargs: Any) -> None:
        pass

    def read_group_motor_position(self) -> np.ndarray:
        # For BaseManipulator compatibility when reading from "robot"
        if not self.is_connected or self.rtde_rec is None:
            return np.ones(len(self.SERVO_IDS)) * np.nan
        try:
            q = np.asarray(self.rtde_rec.getActualQ(), dtype=float)
            # Convert rad -> motor units using BaseManipulator mapping
            if self.config is None:
                return np.ones(len(self.SERVO_IDS)) * np.nan
            return self._radians_vec_to_motor_units(q)
        except Exception as e:
            logger.warning(f"read_group_motor_position failed: {e}")
            return np.ones(len(self.SERVO_IDS)) * np.nan

    def write_group_motor_position(self, q_target: np.ndarray, enable_gripper: bool) -> None:
        # q_target is in motor units per BaseManipulator. For UR, convert to radians and call moveJ
        if not self.is_connected or self.rtde_ctrl is None:
            return
        try:
            q_rad = self._units_vec_to_radians(np.asarray(q_target, dtype=float)).tolist()
            self.rtde_ctrl.moveJ(q_rad, self.speed, self.acc)
        except Exception as e:
            logger.warning(f"write_group_motor_position failed: {e}")

    def set_motors_positions(self, q_target_rad: np.ndarray, enable_gripper: bool = False) -> None:
        # Override to directly command UR and sync simulation
        if self.is_connected and self.rtde_ctrl is not None:
            try:
                self.rtde_ctrl.moveJ(np.asarray(q_target_rad, dtype=float).tolist(), self.speed, self.acc)
            except Exception as e:
                logger.warning(f"moveJ failed: {e}")
        # Always mirror to sim
        super().set_motors_positions(q_target_rad=q_target_rad, enable_gripper=enable_gripper)

    def control_gripper(self, open_command: float, **kwargs: Any) -> None:
        # Open/close Robotiq on UR controller; also mirror to sim if needed (no sim gripper for UR5e)
        open_command = float(np.clip(open_command, 0.0, 1.0))
        if self.is_connected:
            try:
                target = int(open_command * 255)
                self.gripper.move(target, self.gripper_speed, self.gripper_force)
            except Exception as e:
                logger.error(f"Error controlling gripper: {e}")

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        # Match BaseRobot signature used elsewhere in the app
        if not self.is_connected or self.rtde_rec is None:
            # Fall back to sim via BaseManipulator
            return super().get_observation(do_forward=True)
        joints_position = np.asarray(self.rtde_rec.getActualQ(), dtype=float)
        tcp_pose = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=float)
        logger.debug(f"q: {joints_position}, tcp: {tcp_pose}")
        return joints_position, tcp_pose

    async def move_robot_absolute(
        self,
        target_position: np.ndarray,
        target_orientation_rad: Optional[np.ndarray],
        speed_m_s: float = 0.05,
        acc_m_s2: float = 0.2,
        **kwargs: Any,
    ) -> None:
        if not self.is_connected or self.rtde_ctrl is None:
            # Use sim IK + set in simulation
            await super().move_robot_absolute(target_position, target_orientation_rad, **kwargs)
            return
        # Convert orientation to rotation vector (UR expects rotvec)
        if target_orientation_rad is None and self.rtde_rec is not None:
            curr = self.rtde_rec.getActualTCPPose()
            rx, ry, rz = curr[3], curr[4], curr[5]
        else:
            rotvec = R.from_euler("xyz", target_orientation_rad).as_rotvec() if target_orientation_rad is not None else np.zeros(3)
            rx, ry, rz = float(rotvec[0]), float(rotvec[1]), float(rotvec[2])
        x, y, z = [float(v) for v in target_position]
        try:
            self.rtde_ctrl.moveL([x, y, z, rx, ry, rz], float(speed_m_s), float(acc_m_s2))
        except Exception as e:
            logger.warning(f"moveL failed: {e}")

    async def move_robot_relative(
        self,
        target_position: np.ndarray,
        target_orientation_rad: Optional[np.ndarray],
        step_time_s: float = 0.05,
        acc_m_s2: float = 0.5,
    ) -> None:
        if not self.is_connected or self.rtde_ctrl is None or self.rtde_rec is None:
            # Fallback: approximate with BaseManipulator absolute step
            curr_pos, curr_euler = self.forward_kinematics()
            new_pos = curr_pos + np.array([v if v is not None else 0.0 for v in target_position], dtype=float)
            if target_orientation_rad is not None:
                d_euler = np.array([float(v) if (v is not None and not np.isnan(v)) else 0.0 for v in target_orientation_rad], dtype=float)
                new_euler = R.from_euler("xyz", curr_euler) * R.from_euler("xyz", d_euler)
                new_euler = new_euler.as_euler("xyz")
            else:
                new_euler = curr_euler
            await self.move_robot_absolute(new_pos, new_euler)
            return

        curr = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=float)
        pos = curr[:3]
        rotvec = curr[3:6]

        dx, dy, dz = [float(v) if v is not None else 0.0 for v in target_position]
        new_pos = pos + np.array([dx, dy, dz], dtype=float)

        R_curr = R.from_rotvec(rotvec)
        if target_orientation_rad is not None:
            d_euler = np.array([
                float(v) if (v is not None and not np.isnan(v)) else 0.0 for v in target_orientation_rad
            ], dtype=float)
            R_delta = R.from_euler("xyz", d_euler)
            R_new = R_curr * R_delta
        else:
            R_new = R_curr
        new_rotvec = R_new.as_rotvec()

        t = float(np.clip(step_time_s, 0.02, 0.1))
        v_lin = 0.3
        a_lin = float(np.clip(acc_m_s2, 0.2, 1.0))
        pose = [
            float(new_pos[0]),
            float(new_pos[1]),
            float(new_pos[2]),
            float(new_rotvec[0]),
            float(new_rotvec[1]),
            float(new_rotvec[2]),
        ]
        try:
            self.rtde_ctrl.servoL(pose, a_lin, v_lin, t, 0.1, 300)
        except Exception as e:
            logger.warning(f"servoL failed: {e}")

    def status(self) -> RobotConfigStatus:
        return RobotConfigStatus(
            name=self.name,
            device_name=getattr(self, "SERIAL_ID", None),
        )

    async def move_to_initial_position(self) -> None:
        if not self.is_connected or self.rtde_ctrl is None:
            await super().move_to_initial_position(open_gripper=True)
            return
        q_home = [-0.079, -1.98, 2.03, 3.70, -1.58, -4.78]
        try:
            self.rtde_ctrl.moveJ(q_home, 0.4, 0.6)
        except Exception:
            pass
        try:
            self.gripper.move_and_wait_for_pos(self.gripper.get_open_position(), self.gripper_speed, self.gripper_force)
        except Exception:
            pass
        if self.rtde_rec is not None:
            pose = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=float)
            position = pose[:3]
            euler_xyz = R.from_rotvec(pose[3:6]).as_euler("xyz")
            self.initial_position = position
            self.initial_orientation_rad = euler_xyz

    async def move_to_sleep(self) -> None:
        if not self.is_connected or self.rtde_ctrl is None:
            await super().move_to_sleep()
            return
        q_home = [-0.079, -1.98, 2.03, 3.70, -1.58, -4.78]
        try:
            self.rtde_ctrl.moveJ(q_home, 0.4, 0.6)
        except Exception:
            pass

    def forward_kinematics(self, sync_robot_pos: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if self.is_connected and self.rtde_rec is not None:
            pose = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=float)
            position = pose[:3]
            euler_xyz = R.from_rotvec(pose[3:6]).as_euler("xyz")
            return position, euler_xyz
        return super().forward_kinematics(sync_robot_pos=sync_robot_pos)

    def get_info_for_dataset(self) -> BaseRobotInfo:
        num_joints = len(self.SERVO_IDS)
        return BaseRobotInfo(
            robot_type=self.name,
            action=FeatureDetails(
                dtype="float32",
                shape=[num_joints],
                names=[f"motor_{i}" for i in range(num_joints)],
            ),
            observation_state=FeatureDetails(
                dtype="float32",
                shape=[num_joints],
                names=[f"motor_{i}" for i in range(num_joints)],
            ),
        )
